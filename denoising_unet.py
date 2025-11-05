import torch 
import copy
import random 
import torch.nn.functional as F 
from functools import partial 
from typing import Any, Tuple, Optional, Callable, List 
from inspect import isfunction 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
import math
from attend import Attend

def exists(x: Any) -> bool:
    return x is not None 

def default(val: Any, d: Any):
    if exists(val):
        return val 
    return d() if isfunction(d) else d 

def cast_tuple(t: Any, length: int = 1):
    if isinstance(t, tuple):
        return t 
    return ((t,) * length)

def divisible_by(numer: int, denom: int):
    return (numer % denom) == 0 

def identity(t: Any, *args, **kwargs):
    return t 

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def cycle(dl):
    while True:
        for data in dl:
            yield data 

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5 

def Upsample(dim: int, dim_out: Optional[int] = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=dim, out_channels=default(dim_out, dim), kernel_size=3, padding = 1)
    )

def Downsample(dim: int, dim_out: Optional[int] = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(in_channels=dim * 4, out_channels=default(dim_out, dim), kernel_size=1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(data=torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim = 1) * self.g * self.scale

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim: int , is_random: bool = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(data=torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, dropout: float = 0.) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
        self.norm = RMSNorm(dim=dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scale_shift = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None, dropout: float = 0.) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dim, out_features=dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim=dim, dim_out=dim_out, dropout=dropout)
        self.block2 = Block(dim=dim_out, dim_out=dim_out)
        self.res_conv = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4
    ) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim=dim)

        self.mem_kv = nn.Parameter(data=torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
            RMSNorm(dim=dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        flash: bool = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads 

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(data=torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class UNet(nn.Module):
    def __init__(
        self,
        dim: int ,
        num_classes: Optional[int] = None,
        init_dim: int = None,
        out_dim: int = None,
        cond_dim: int = 384,
        dim_mults: Tuple = (1, 2, 4, 8),
        channels: int = 2,
        self_condition: bool = False,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10000,
        dropout: float = 0.,
        attn_dim_head: int = 8,
        attn_heads: int = 4,
        full_attn: bool = None,    # defaults to full attention only for inner most layer
        flash_attn: bool = False,
        condition_type: str = None,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.condition_type = condition_type
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(in_channels=input_channels, out_channels=init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Only create label embedding if num_classes is provided
        if num_classes is not None:
            self.label_embed = nn.Embedding(num_classes, time_dim)
        else:
            self.label_embed = None

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor = None, x_self_cond: torch.Tensor = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) 
        if cond is not None:
            if not isinstance(cond, torch.Tensor):
                cond = torch.tensor(cond, dtype=torch.float32)
            if self.condition_type == "class":
                if self.label_embed is not None:
                    t = t + self.label_embed(cond)
            elif self.condition_type == "latent":
                t = t + self.cond_mlp(cond)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)