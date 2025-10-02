import numpy as np
import torch 
import math 
import copy 
import random 
import enum
from lightly.models.utils import set_at_index, mask_at_index
from torch import nn 
from skimage import measure
import torch.nn.functional as F 
from typing import List, Callable, Optional, Dict, List, Tuple
from denoising_unet import UNet
import copy
from lightning.pytorch.core import LightningModule

"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DDPM(LightningModule):
    def __init__(
        self,
        denoising_model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "sigmoid",
        auto_normalize: bool = True,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: int = 5,
        model_var_type: ModelVarType = ModelVarType.FIXED_SMALL,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        loss_type: LossType = LossType.MSE,
        rescale_timesteps: bool = False,
        condition_type: Optional[str] = None,
        latent_encoder: Optional[nn.Module] = None,
        concat_segmentation: bool = False,
        task_network: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
    
        self.latent_encoder = latent_encoder

        self.condition_type = condition_type
        self.rescale_timesteps = rescale_timesteps
        self.model_var_type = model_var_type
        self.model_mean_type = model_mean_type 
        self.loss_type = loss_type
        self.concat_segmentation = concat_segmentation
        self.model = denoising_model 
        if self.latent_encoder is not None:
            for p in self.latent_encoder.parameters():
                p.requires_grad = False
        
        self.task_network = task_network 
        if self.task_network is not None:
            for p in self.task_network.parameters():
                p.requires_grad = False

        self.T = timesteps 
        self.channels = self.model.channels 
        betas = get_named_beta_schedule(schedule_name=beta_schedule, num_diffusion_timesteps=self.T)
        self.betas = betas 
        assert len(betas.shape) == 1 
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.T,)

        # q(x_t, x_{t-1})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def q_mean_variance(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get the distribution q(x_t, x_0)
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_0.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance 

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Diffuse the data, i.e., sample from q(x_t, x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape 
        return(
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        if not self.concat_segmentation:
            assert x_0.shape == x_t.shape 
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_0
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        if not self.concat_segmentation:
            assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_0.shape[0]
            )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, cond=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = self.model(x, self._scale_timesteps(t), cond=cond, **model_kwargs)
        if C == 2:
            x = x[:, [0], :, :]

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_x0 = process_xstart(
            self._predict_x0_from_eps(x_t=x, t=t, eps=model_output)
        )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_0=pred_x0, x_t=x, t=t
        )
        if not self.concat_segmentation:
            assert (model_mean.shape == model_log_variance.shape == pred_x0.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": pred_x0,
        }

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        if not self.concat_segmentation:
            assert x_t.shape == eps.shape 
        else:
            B, C = x_t.shape[:2]
            if C == 2:
                x_t = x_t[:, [0], :, :]
            assert x_t.shape == eps.shape 
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t 
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_x0(self, x_t, t, pred_x0):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_x0
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable] = None,
        model_kwargs=None,
        cond=None,
    ) -> torch.Tensor:
        """
        Sample x_{t-1}
        """

        out = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cond=cond,
        )
        noise = torch.randn_like(x[:, [0], :, :])
        nonzero_mask = (
            (t!=0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x0": out["pred_x0"]}

    def p_sample_loop(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        cond=None,
        segmentation=None,
    ):
        """
        Generate samples from the model.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            cond=cond,
            segmentation=segmentation,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        cond=None,
        segmentation=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.T))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, desc="Iterative sampling...")

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            if segmentation is not None:
                img = torch.cat((img, segmentation), axis=1)
            with torch.no_grad():
                out = self.p_sample(
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    cond=cond,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            t_prev: torch.Tensor, 
            eta: float = 0.0,
            cond: Optional[torch.Tensor] = None, 
            model_kwargs: Optional[Dict] = None, 
            clip_denoised: bool = True, 
            denoised_fn: Optional[Callable] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if model_kwargs is None:
            model_kwargs = {}

        # as param to p_mean_variance, x has two channels (image and segmentation) that are passed along to the UNet
        out = self.p_mean_variance(x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs, cond=cond)
        # The U-Net predicts only one channel (`out` variable above), so we need to index the img channel from x before performing the ddim updates.
        if x.shape[1] == 2:
            x = x[:, [0], :, :]
        pred_x0 = out["pred_x0"]
        # alphas for current and previous t's
        alphas_cumprod = torch.from_numpy(self.alphas_cumprod).to(x.device, x.dtype)
        a_t = alphas_cumprod[t].view(-1, *([1] * (x.dim() - 1)))
        a_prev = alphas_cumprod[t_prev].view(-1, *([1] * (x.dim() - 1)))
    

        eps = (x - a_t.sqrt() * pred_x0) / (1 - a_t).sqrt()
        # DDIM update 
        sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
        noise = torch.randn_like(x) if eta > 0 else 0.0
        mean_pred = pred_x0 * a_prev.sqrt() + eps * (1 - a_prev).sqrt()
        x_prev = mean_pred + sigma * noise
        return x_prev, pred_x0

    def ddim_sample_loop(
            self, 
            shape: Tuple[int, int, int, int], 
            num_steps: int= 100, 
            eta: float = 0.0, 
            cond: Optional[torch.Tensor] = None, 
            model_kwargs: Optional[Dict] = None, 
            device: Optional[torch.device] = None, 
            progress: bool = False, 
            segmentation: Optional[torch.Tensor] = None, 
            clip_denoised: bool = True, 
            denoised_fn: Optional[Callable] = None, 
            noise: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        if device is None:
            device = next(self.model.parameters()).device
        if model_kwargs is None:
            model_kwargs = {}
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        
        # time schedule with num_steps between 0 and T = 1000
        times = np.linspace(self.T - 1, 0, num_steps, dtype=int)
        times_next = np.append(times[1:], 0)

        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(zip(times, times_next), total=num_steps, desc="DDIM sampling")
        else:
            iterator = zip(times, times_next)
        for t_i, t_prev_i in iterator:
            t = torch.full((shape[0],), t_i, device=device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_prev_i, device=device, dtype=torch.long)
            if segmentation is not None:
                img_in = torch.cat((img, segmentation), axis=1)
            else:
                img_in = img
            with torch.no_grad():
                img, _ = self.ddim_sample(img_in, t, t_prev, eta=eta, cond=cond, model_kwargs=model_kwargs, clip_denoised=clip_denoised, denoised_fn=denoised_fn)
        return img

    def _prior_bpd(self, x_0):
        """
        Get the prior KL divergence.
        """
        batch_size = x_0.shape[0]
        t = torch.tensor([self.T - 1] * batch_size, device=x_0.device)
        mean, _, log_variance = self.q_mean_variance(x_0=x_0, t=t)
        kl_prior = normal_kl(mean1=mean, logvar1=log_variance, mean2=0.0, logvar2=0.0)
        kl_prior = mean_flat(kl_prior) / np.log(2.0)
        return kl_prior

    def _vb_terms_bpd(
        self, x_0, x_t, t, cond=None, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_x0': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_0=x_0, x_t=x_t, t=t
        )
        out = self.p_mean_variance( x_t, t, cond=cond, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_0.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_x0": out["pred_x0"]}

    def forward(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        cond: torch.Tensor = None, 
        segmentation: torch.Tensor = None, 
        ground_truth: torch.Tensor = None,
        model_kwargs: Dict = None, 
        noise: torch.Tensor = None, 
        verbose: bool = True, 
        ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)

        if self.concat_segmentation:
            assert segmentation is not None
            x_t = torch.cat((x_t, segmentation), axis=1)

        # latent_encoding = self.latent_encoder(x0)
        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                x_0=x_0,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = self.model(x_t, self._scale_timesteps(t), cond=cond, **model_kwargs)
            pred_x0 = self._predict_x0_from_eps(x_t=x_t, t=t, eps=model_output)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                if not self.concat_segmentation:
                    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    x_0=x_0,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_0=x_0, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_0,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            if not self.concat_segmentation:
                assert model_output.shape == target.shape == x_0.shape
            
            # Compute MSE loss
            mse_loss = (target - model_output) ** 2
            terms["mse"] = mean_flat(mse_loss)
                
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

            if self.task_network is not None:
                task_output = self.task_network(pred_x0) 
                terms["task_loss"] = mean_flat(ground_truth - task_output)  
                terms["loss"] += terms["task_loss"]
            else: 
                terms["task_loss"] = None
        else:
            raise NotImplementedError(self.loss_type)

        return terms, {
            "pred": model_output,
            "x_t": x_t,
            "pred_x0": self._predict_x0_from_eps(x_t=x_t, t=t, eps=model_output),
            "target": noise,
        }

    def training_step(self, batch, batch_idx):
        imgs, segmentations, ground_truth, _ = batch 
        if self.task_network is None:
            ground_truth = None
        device = imgs.device
        self.imgs = imgs
        self.segmentations = segmentations

        # Move to device and ensure contiguous tensors
        imgs = imgs.contiguous()
        if segmentations is not None:
            segmentations = segmentations.contiguous()
    
        if self.condition_type == "latent":
            latent_code, context = self.latent_encoder.forward_features(imgs, return_patches=True)
            cond = latent_code
        else: 
            cond = None

        # Condition that gets concatenated to input noise (spatial)
        if not self.concat_segmentation:
            segmentations = None

        t = torch.randint(0, 1000, (imgs.shape[0],), device=device).long()
        losses, model_outputs = self(x_0=imgs, t=t, cond=cond, segmentation=segmentations, ground_truth=ground_truth)
        loss = losses["loss"].mean()
    
        # self.log("Loss/mean", loss, on_epoch=True, sync_dist=True)
        # self.log("Loss/min", losses["loss"].min(), on_epoch=True, reduce_fx=torch.min, sync_dist=True)
        # self.log("Loss/max", losses["loss"].max(), on_epoch=True, reduce_fx=torch.max, sync_dist=True)
        
        # Ensure loss is on the correct device and requires grad
        if not loss.requires_grad:
            loss = loss.requires_grad_(True)
            
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.9, 0.99))
        return [optimizer]
 
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

