import tifffile 
import numpy as np 
import glob 
from data.stedfm_dendrites_dataset import DendriticFActinDataset 
from stedfm.DEFAULTS import BASE_PATH 
import matplotlib.pyplot as plt 
import os 
from stedfm import get_pretrained_model_v2 
from tiffwrapper import make_composite 
from skimage.filters import threshold_otsu
from models.UNet import UNet 
from denoising_unet import UNet as DenoisingUNet 
from diffusion_model import DDPM
from typing import List, Dict
import torch 
from torch import nn 
from stedfm.decoders import get_decoder 
from diffusion_model import DDPM 
from stedfm.configuration import Configuration 
import argparse 
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints-dir", type=str, default="/home-local/Frederic/baselines/SR-baselines")
parser.add_argument("--dataset", type=str, default="DendriticFActinDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets/DendriticFActinDataset"))
parser.add_argument("--unet-checkpoint", type=str, default="/home/frederic/TA-GAN/checkpoints/UNet_Dendrites/params.net")
parser.add_argument("--rings-threshold", type=float, default=0.25)
parser.add_argument("--fibres-threshold", type=float, default=0.4)
args = parser.parse_args()


def load_unet():
    unet = UNet(in_channels=1, out_channels=2)
    unet.load_state_dict(torch.load(args.unet_checkpoint))
    return unet

def load_DM():
    latent_encoder, cfg = get_pretrained_model_v2(
        name="mae-lightning-small",
        weights="MAE_SMALL_STED",
        as_classifier=True,
    )
    denoising_model = DenoisingUNet(
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

    ckpt = torch.load(os.path.join(args.checkpoints_dir, "DM_199.pth"), map_location="cpu", weights_only=False)
    diffusion_model.load_state_dict(ckpt["model"], strict=True)
    return diffusion_model

def check_ground_truth(segmentation: np.ndarray, threshold: int = 2500):
    return np.count_nonzero(segmentation) >= threshold


def load_data(path: str, crop_size: int = 224):
    files = glob.glob(f"{path}/test/*.tif")
    print(f"Found {len(files)} test images")
    data = []

    for i, f in enumerate(files):
        image_data = tifffile.imread(f)
        confocal = image_data[0] / 255.
        sted = image_data[1] / 255.
        rings = image_data[2]
        fibres = image_data[3]
        if rings.max() > 0:
            rings = (rings / rings.max()).astype(np.uint8)
        if fibres.max() > 0:
            fibres = (fibres / fibres.max()).astype(np.uint8)
    
        good_crop = False 
        while not good_crop:
            y = np.random.randint(0, sted.shape[0] - crop_size)
            x = np.random.randint(0, sted.shape[1] - crop_size)
            sted_crop = sted[y:y+crop_size, x:x+crop_size]
            confocal_crop = confocal[y:y+crop_size, x:x+crop_size]
            rings_crop = rings[y:y+crop_size, x:x+crop_size]
            fibres_crop = fibres[y:y+crop_size, x:x+crop_size]
            good_crop = check_ground_truth(rings_crop) or check_ground_truth(fibres_crop)      
        data.append((os.path.basename(f), confocal_crop, sted_crop, rings_crop, fibres_crop, y, x))
    return data

def filter_files(files: List[str], model: str):
    if "Pix2Pix" in model:
        files = [f for f in files if "fake_B" in f]
    else:
        files = [f for f in files if "fakeSTED" in f]
    files = [f for f in files if "seg" not in f]
    return files

def binarize(segmentation: np.ndarray):
    thresholds = {0: args.rings_threshold, 1: args.fibres_threshold}
    new_segmentation = np.zeros_like(segmentation)
    for ch in range(segmentation.shape[0]):
        new_segmentation[ch] = (segmentation[ch] >= thresholds[ch]).astype(np.uint8)
    return new_segmentation
            

def compute_dice(ground_truth: np.ndarray, prediction: np.ndarray):
    dice_scores = []
    for ch in range(prediction.shape[0]):
        gt_ch = ground_truth[ch]
        if np.count_nonzero(gt_ch) == 0:
            dice_scores.append(-1)
        else:
            pred_ch = prediction[ch]
            intersection = np.logical_and(pred_ch, gt_ch)
            dice = (2 * intersection.sum()) / (pred_ch.sum() + gt_ch.sum())
            dice_scores.append(dice)
    return dice_scores

def plot_results(results: Dict, metric_key: str):
    colors = {"Pix2Pix_2025-10-06": "tab:green", "TAGANDendriticFactin_2025-10-03": "dodgerblue", "LALAGAN_2025-10-08": "#CC503E", "DM": "fuchsia"}
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for pos, model_key in enumerate(results.keys()):
        data = results[model_key]
        x = np.random.normal(loc=pos, scale=0.05, size=len(data))
        ax.scatter(x, data, label=model_key, color=colors[model_key])
        boxes = ax.boxplot(data, positions=[pos], showmeans=True, patch_artist=True,
                           meanline=True, meanprops=dict(color='black', linewidth=1.5),
                           medianprops=dict(linewidth=0),
                           boxprops=dict(facecolor='none'))
        
    ax.legend()
    ax.set_xlabel("Model")
    ax.set_xticks([])
    ax.set_ylabel(metric_key)
    plt.savefig(f"./tmp/figures/{metric_key}.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

    
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS = ["Pix2Pix_2025-10-06", "TAGANDendriticFactin_2025-10-03", "LALAGAN_2025-10-08"]
    print(len(MODELS))

    unet = load_unet().to(DEVICE)
    unet.eval()

    diffusion_model = load_DM().to(DEVICE)
    diffusion_model.eval()

    inference_paths = [os.path.join(args.checkpoints_dir, "results", model, "test_200" if model != "LALAGAN_2025-10-08" else "test_300", "images") for model in MODELS]
    print(len(inference_paths))

    results = {
        "PSNR": {key: [] for key in MODELS + ["DM"]},
        "MSE": {key: [] for key in MODELS + ["DM"]},
        "MSSIM": {key: [] for key in MODELS + ["DM"]},
        "Rings_dice": {key: [] for key in MODELS + ["DM"]},
        "Fibres_dice": {key: [] for key in MODELS + ["DM"]},
    }

    data = load_data(path=args.dataset_path)
    
    for i, (fname, confocal, sted, rings, fibres, y, x) in tqdm(enumerate(data), total=len(data), desc="Evaluating models..."):
        


        assert sted.shape == (224, 224) == confocal.shape == rings.shape == fibres.shape
        gt = np.stack([rings, fibres])
        gt_rgb = make_composite(gt, luts=['green', 'magenta'], ranges=[(0, 1), (0, 1)])
        fig, axs = plt.subplots(2, len(MODELS) + 2)
        axs[0, 0].imshow(sted, cmap="hot", vmax=sted.max())
        axs[1, 0].imshow(gt_rgb)

        torch_confocal = torch.from_numpy(confocal).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            sample = diffusion_model.p_sample_loop(
                shape=(1, 1, 224, 224),
                cond=None,
                segmentation=torch_confocal,
                progress=True,
            )
            
            dm_seg = unet(sample).squeeze().cpu().numpy()
            sample_numpy = sample.squeeze().cpu().numpy()
            dm_seg = binarize(dm_seg)
            dm_seg_rgb = make_composite(dm_seg, luts=['green', 'magenta'], ranges=[(0, 1), (0, 1)])

        axs[0, -1].imshow(sample_numpy, cmap="hot", vmax=sted.max())
        axs[1, -1].imshow(dm_seg_rgb)
        axs[0, -1].set_title("DM")
        
        psnr = peak_signal_noise_ratio(sted, sample_numpy)
        mse = normalized_root_mse(sted, sample_numpy)
        mssim_object = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
        torch_sted = torch.from_numpy(sted).unsqueeze(0).unsqueeze(0).to(DEVICE).float() 
        mssim = mssim_object(torch_sted, sample).item()
        results["PSNR"]["DM"].append(psnr)
        results["MSE"]["DM"].append(mse)
        results["MSSIM"]["DM"].append(mssim)

        dice_scores = compute_dice(gt, dm_seg)
        if dice_scores[0] != -1:
            results["Rings_dice"]["DM"].append(dice_scores[0])
        if dice_scores[1] != -1:
            results["Fibres_dice"]["DM"].append(dice_scores[1])

        axs[1, -1].set_title(f"{dice_scores[0]:.2f}, {dice_scores[1]:.2f}")
        print(fname)
        for j, (model, path) in enumerate(zip(MODELS, inference_paths)):
            inference_files = glob.glob(f"{path}/*.tif") 
            inference_files = filter_files(files=inference_files, model=model)
            
            assert len(inference_files) == 26, f"Expected 26 files, got {len(inference_files)} for {model}"
            
                
            found = [fname.split(".")[0] == f.split("/")[-1].split(".")[0] for f in inference_files]
            matching_file = inference_files[found.index(True)]
            print(matching_file)
            
            large_image = tifffile.imread(matching_file).squeeze() / 255.
            pred_img = large_image[y:y+224, x:x+224]
            assert pred_img.shape == (224, 224)
            
            torch_img = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
            with torch.no_grad():
                pred_seg = unet(torch_img).squeeze().cpu().numpy()
                pred_seg = binarize(pred_seg)
                pred_seg_rgb = make_composite(pred_seg, luts=['green', 'magenta'], ranges=[(0, 1), (0, 1)])

            axs[0, j+1].imshow(pred_img, cmap="hot", vmax=sted.max())
            axs[1, j+1].imshow(pred_seg_rgb) 

            psnr = peak_signal_noise_ratio(sted, pred_img)
            mse = normalized_root_mse(sted, pred_img)
            mssim_object = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
            torch_sted = torch.from_numpy(sted).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
            mssim = mssim_object(torch_sted, torch_img).item()
            results["PSNR"][model].append(psnr)
            results["MSE"][model].append(mse)
            results["MSSIM"][model].append(mssim)

            dice_scores = compute_dice(gt, pred_seg)
            if dice_scores[0] != -1:
                results["Rings_dice"][model].append(dice_scores[0])
            if dice_scores[1] != -1:
                results["Fibres_dice"][model].append(dice_scores[1])

            axs[1, j+1].set_title(f"{dice_scores[0]:.2f}, {dice_scores[1]:.2f}")
        print("\n")
        for ax in axs.ravel():
            ax.axis("off")
        axs[0, 0].set_title("STED")
        axs[0, 1].set_title("Pix2Pix")
        axs[0, 2].set_title("TAGAN")
        axs[0, 3].set_title("LALAGAN")
        axs[0, -1].set_title("DM")
        fig.savefig(f"./tmp/samples/sample_{i}.pdf", pad_inches=0.1, dpi=1200, bbox_inches='tight')
        plt.close(fig)


    for metric_key in results.keys():
        plot_results(results[metric_key], metric_key)

        
if __name__=="__main__":
    main()