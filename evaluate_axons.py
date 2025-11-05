import tifffile 
import numpy as np 
import glob 
from data.stedfm_axons_dataset import AxonalRingsDataset
from stedfm.DEFAULTS import BASE_PATH 
import matplotlib.pyplot as plt 
import os 
from models.UNet import UNet  
from typing import List, Dict
import torch 
from torch import nn 
import argparse 
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints-dir", type=str, default="/home-local/Frederic/baselines/SR-baselines")
parser.add_argument("--dataset", type=str, default="AxonalRingsDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets/AxonalRingsDataset"))
parser.add_argument("--unet-checkpoint", type=str, default="/home/frederic/TA-GAN/checkpoints/UNet_Axons/axon-pretrained-unet/params.net")
parser.add_argument("--rings-threshold", type=float, default=0.02)
parser.add_argument("--epoch", type=int, default=300)
args = parser.parse_args()

def load_unet():
    unet = UNet(in_channels=1, out_channels=2)
    unet.load_state_dict(torch.load(args.unet_checkpoint))
    return unet

def load_test_data(path: str):
    files = glob.glob(f"{path}/test/*.tif")
    data = [] 
    for i, f in enumerate(files):
        image_data = tifffile.imread(f)
        confocal, sted, rings = image_data[0] / 255., image_data[1] / 255., image_data[2] / 255. 
        data.append((os.path.basename(f), confocal, sted, rings)) 
    return data


def filter_files(files: List[str], model: str):
    if "Pix2Pix" in model:
        files = [f for f in files if "fake_B" in f]
    else:
        files = [f for f in files if "fakeSTED" in f]
    files = [f for f in files if "seg" not in f]
    return files


def compute_dice(ground_truth: np.ndarray, prediction: np.ndarray):
    intersection = np.logical_and(prediction, ground_truth)
    dice = (2 * intersection.sum()) / (prediction.sum() + ground_truth.sum())
    return dice

def plot_results(results: Dict, metric_key: str):
    os.makedirs("./tmp/AxonalRings/figures", exist_ok=True)
    colors = {"Pix2Pix": "tab:green", "TAGAN": "dodgerblue", "SAGAN": "fuchsia"}
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for pos, model_key in enumerate(results.keys()):
        data = results[model_key]
        x = np.random.normal(loc=pos, scale=0.05, size=len(data))
        ax.scatter(x, data, label=model_key, color=colors[model_key], alpha=0.7)
        boxes = ax.boxplot(data, positions=[pos], showmeans=True, patch_artist=True,
                           meanline=True, meanprops=dict(color='black', linewidth=1.5),
                           medianprops=dict(linewidth=0),
                           boxprops=dict(facecolor='none'))
        
    ax.legend()
    ax.set_xlabel("Model")
    ax.set_xticks([])
    ax.set_ylabel(metric_key)
    plt.savefig(f"./tmp/AxonalRings/figures/{metric_key}.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    os.makedirs("./tmp/AxonalRings", exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = load_unet().to(DEVICE)
    unet.eval()

    data = load_test_data(args.dataset_path)
    
    MODELS = ["Pix2PixAxonalFactin_2025-11-04", "TAGANAxonalFactin_2025-11-04", "SAGANAxonalFactin_2025-11-05"]
    inference_paths = [
        os.path.join(args.checkpoints_dir, "results", model, f"test_{args.epoch}", "images") for model in MODELS
    ]

    model_names = [model.split("AxonalFactin")[0] for model in MODELS]

    results = {
        "PSNR": {key: [] for key in model_names},
        "MSE": {key: [] for key in model_names},
        "MSSIM": {key: [] for key in model_names},
        "Dice": {key: [] for key in model_names},
    }

    for i, (fname, confocal, sted, rings) in tqdm(enumerate(data), total=len(data), desc="Evaluating models..."):
        fig, axs = plt.subplots(2, len(MODELS) + 1)
        axs[0, 0].imshow(sted, cmap="hot", vmax=sted.max())
        axs[1, 0].imshow(rings, cmap="gray", vmax=1)
        
        for j, (model, path) in enumerate(zip(MODELS, inference_paths)):
            model_name = model.split("AxonalFactin")[0]
            inference_files = glob.glob(f"{path}/*.tif")
            inference_files = filter_files(inference_files, model_name)

            if model_name == "Pix2Pix":
                temp_inference_files = [os.path.basename(f).replace("_fake_B.tif", "") for f in inference_files]
            else:
                temp_inference_files = [os.path.basename(f).replace("_fakeSTED.tif", "") for f in inference_files]

            found = [fname.split(".")[0] == f.split("/")[-1].split(".")[0] for f in temp_inference_files]
            matching_file = inference_files[found.index(True)]
            
            pred = tifffile.imread(matching_file).squeeze() / 255.
            torch_pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(DEVICE).float() 
            with torch.no_grad():
                pred_seg = unet(torch_pred).squeeze().cpu().numpy() > args.rings_threshold
                pred_seg = pred_seg[1]

            psnr = peak_signal_noise_ratio(rings, pred)
            mse = normalized_root_mse(sted, pred)
            mssim_object = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction="none") 
            torch_sted = torch.from_numpy(sted).unsqueeze(0).unsqueeze(0).to(DEVICE).float() 
            mssim = mssim_object(torch_sted, torch_pred).item() 

            results["PSNR"][model_name].append(psnr)
            results["MSE"][model_name].append(mse)
            results["MSSIM"][model_name].append(mssim)

            if np.count_nonzero(rings) == 0 and np.count_nonzero(pred_seg) == 0:
                dice = 1.0 
                results["Dice"][model_name].append(dice)
            elif np.count_nonzero(rings) == 0 and np.count_nonzero(pred_seg) != 0:
                dice = 0.0
            else:
                dice = compute_dice(rings, pred_seg)
                results["Dice"][model_name].append(dice)

            axs[0, j+1].imshow(pred, cmap="hot", vmax=pred.max())
            axs[1, j+1].imshow(pred_seg, cmap="gray", vmax=1)
            axs[1, j+1].set_title(f"Dice: {dice:.2f}")
            
            axs[1, j+1].set_title(f"Dice: {dice:.2f}")

        for ax in axs.ravel():
            ax.axis("off")
        axs[0, 0].set_title("STED")
        axs[0, 1].set_title("Pix2Pix")
        axs[0, 2].set_title("TAGAN")
        axs[0, 3].set_title("SAGAN")
        fig.savefig(f"./tmp/AxonalRings/sample_{i}.png", pad_inches=0.1, dpi=1200, bbox_inches='tight')
        plt.close(fig)

    for metric_key in results.keys():
        plot_results(results[metric_key], metric_key)

            
                