import numpy as np 
import torch 
from diffusion_model import DDPM 
from stedfm.DEFAULTS import BASE_PATH 
from stedfm import get_pretrained_model_v2
from torch.utils.data import DataLoader 
import os 
import glob 
from tqdm import tqdm, trange 
from denoising_unet import UNet 
from data.stedfm_dendrites_dataset import DendriticFActinDataset
from torch.utils.data import DataLoader
import tifffile


if __name__=="__main__":
    os.makedirs("./results/DM/test_100/images", exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = glob.glob(f"{BASE_PATH}/Datasets/DendriticFActinDataset/valid/*.tif")
    print(f"[---] Found {len(files)} training files [---]")
    dataset = DendriticFActinDataset(files)
    dataset = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)

    latent_encoder, cfg = get_pretrained_model_v2(
        name="mae-lightning-small",
        weights="MAE_SMALL_STED",
        blocks="all",
        path=None,
        mask_ratio=0.0,
        pretrained=False,
        as_classifier=True,
        num_classes=4
    )
    latent_encoder.eval()
    latent_encoder.to(DEVICE) 

    denoising_model = UNet(
        dim=64, 
        channels=2,
        out_dim=1,
        cond_dim=cfg.dim,
        dim_mults=(1,2,4),
        condition_type=None,
        num_classes=4
    )

    diffusion_model = DDPM(
        denoising_model=denoising_model, 
        timesteps=1000,
        beta_schedule="linear",
        condition_type=None,
        latent_encoder=latent_encoder,
        concat_segmentation=True,
        task_network=None
    )

    ckpt = "/home-local/Frederic/baselines/TADM/DM_99.pth"
    diffusion_model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"], strict=True)
    diffusion_model.to(DEVICE)
    diffusion_model.eval()

    for batch in tqdm(dataset, desc="... Testing ..."):
        confocal = batch["confocal"]
        sted = batch["STED"]
        filename = batch["image_paths"][0]
        confocal = confocal.to(DEVICE)
        sted = sted.to(DEVICE)
        with torch.no_grad():
            sample = diffusion_model.p_sample_loop(
                shape=(1, 1, confocal.shape[2], confocal.shape[3]),
                cond=None,
                segmentation=confocal,
                progress=True,
            )
        sample = sample.squeeze().cpu().numpy()
        tifffile.imwrite(f"./results/DM/test_100/images/{filename.split('/')[-1]}", sample)
        
å