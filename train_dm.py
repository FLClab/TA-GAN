import numpy as np 
import tifffile
import torch 
import matplotlib.pyplot as plt
import argparse 
from stedfm.DEFAULTS import BASE_PATH
from stedfm import get_pretrained_model_v2
from torch.utils.data import DataLoader
import os
from tqdm import tqdm, trange
import glob
from denoising_unet import UNet
from task_unet import TaskUNet
from diffusion_model import DDPM
import sys 
sys.path.insert(0, "../")
from datasets import DendriticFActinDataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets/DendriticFActinDataset"))
parser.add_argument("--num-epochs", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--task-network", action="store_true")
args = parser.parse_args()

def dm_validation_step(model, valid_dataset: DendriticFActinDataset, device: torch.device):
    indices = np.random.choice(len(valid_dataset), size=5, replace=False)
    model.eval()
    os.makedirs("./DM_samples", exist_ok=True)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            confocal, sted, _, _ = valid_dataset[idx]
            confocal = confocal.unsqueeze(0).to(device)
            sample = model.p_sample_loop(
                shape=(1, 1, confocal.shape[2], confocal.shape[3]),
                cond=None,
                segmentation=confocal,
                progress=True,
            )
            
            sample = sample.squeeze().cpu().numpy()
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(confocal.squeeze().cpu().numpy(), cmap="hot", vmin=0, vmax=1)
            axs[1].imshow(sample, cmap="hot", vmin=0, vmax=1)
            axs[2].imshow(sted.squeeze().cpu().numpy(), cmap="hot", vmin=0, vmax=1)
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(f"./DM_samples/sample_{i}.png", dpi=1200, bbox_inches="tight")
            plt.close(fig)

def tadm_validation_step(model, valid_dataset: DendriticFActinDataset, device: torch.device):
    indices = np.random.choice(len(valid_dataset), size=5, replace=False)
    model.eval()
    os.makedirs("./TA-DM_samples", exist_ok=True)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            confocal, sted, ground_truth,_ = valid_dataset[idx]
            confocal = confocal.unsqueeze(0).to(device)
            sample = model.p_sample_loop(
                shape=(1, 1, confocal.shape[2], confocal.shape[3]),
                cond=None,
                segmentation=confocal,
                progress=True,
            )
            seg_pred = model.task_network(sample)
            seg_pred = seg_pred.squeeze().cpu().numpy() 
            confocal = confocal.squeeze().cpu().numpy()
            sted = sted.squeeze().cpu().numpy() 
            ground_truth = ground_truth.squeeze().cpu().numpy() 
            fig, axs = plt.subplots(2, 4, figsize=(16, 8))
            axs[0, 0].imshow(confocal, cmap="hot", vmin=0, vmax=1)
            axs[0, 0].set_title("Confocal")
            axs[0, 1].imshow(sted, cmap="hot", vmin=0, vmax=1)
            axs[0, 1].set_title("STED")
            axs[0, 2].imshow(sample.squeeze().cpu().numpy(), cmap="hot", vmin=0, vmax=1)
            axs[0, 2].set_title("TA-DM")
            axs[1, 0].imshow(ground_truth[0], cmap="gray", vmin=0, vmax=1)
            axs[1, 0].set_title("GT Rings")
            axs[1, 1].imshow(ground_truth[1], cmap="gray", vmin=0, vmax=1)
            axs[1, 1].set_title("GT Fibers")
            axs[1, 2].imshow(seg_pred[0], cmap="gray", vmin=0, vmax=1)
            axs[1, 2].set_title("TA-DM Rings")
            axs[1, 3].imshow(seg_pred[1], cmap="gray", vmin=0, vmax=1)
            axs[1, 3].set_title("TA-DM Fibers")
            for ax in axs.flatten():
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(f"./TA-DM_samples/sample_{i}.png", dpi=1200, bbox_inches="tight")
            plt.close(fig)


if __name__=="__main__":
    os.makedirs("/home-local/Frederic/baselines/SR-baselines", exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = glob.glob(os.path.join(args.dataset_path, "train", "*.tif"))
    dataset = DendriticFActinDataset(files=files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    valid_files = glob.glob(os.path.join(args.dataset_path, "valid", "*.tif"))
    valid_dataset = DendriticFActinDataset(files=valid_files)

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
    
    task_network = None 
    if args.task_network:
        task_network = TaskUNet(in_channels=1, out_channels=2) 
        task_ckpt = torch.load(f"./pretrained/params.net") 
        task_network.load_state_dict(task_ckpt, strict=True)

    diffusion_model = DDPM(
        denoising_model=denoising_model, 
        timesteps=1000,
        beta_schedule="linear",
        condition_type=None,
        latent_encoder=latent_encoder,
        concat_segmentation=True,
        task_network=task_network
    )

    diffusion_model.to(DEVICE)
    diffusion_model.train()
    model_kwargs = {}

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
    model_name = "TADM" if args.task_network else "DM"
    for epoch in trange(args.num_epochs, desc="... Training ..."):
        diffusion_model.train()
        for batch in tqdm(dataloader, desc="... Batches ..."):
            confocal, sted, ground_truth,_ = batch
            if not args.task_network: 
                ground_truth = None
            else:
                ground_truth = ground_truth.to(DEVICE)
            confocal = confocal.to(DEVICE)
            sted = sted.to(DEVICE)
            t = torch.randint(0, 1000, (sted.shape[0],), device=DEVICE).long()

            optimizer.zero_grad()
            losses, model_outputs = diffusion_model(x_0=sted, t=t, cond=None, segmentation=confocal, ground_truth=ground_truth,model_kwargs=model_kwargs)
            loss = losses["loss"].mean()
            loss.backward()
            optimizer.step()

        if args.task_network:
            tadm_validation_step(diffusion_model, valid_dataset, DEVICE)
        else:
            dm_validation_step(diffusion_model, valid_dataset, DEVICE)
        model_checkpoint =  {
                    "model": diffusion_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
     
        torch.save(model_checkpoint, f"/home-local/Frederic/baselines/SR-baselines/{model_name}.pth")
        if (epoch + 1) % 50 == 0:
            torch.save(model_checkpoint, f"/home-local/Frederic/baselines/SR-baselines/{model_name}_{epoch}.pth")
        scheduler.step()

            