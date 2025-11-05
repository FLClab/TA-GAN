import tifffile 
import numpy as np 
from matplotlib.patches import Patch
import glob 
from data.synaptic_protein_dataset import SynapticProteinDataset 
from stedfm.DEFAULTS import BASE_PATH 
import matplotlib.pyplot as plt 
import os 
from stedfm import get_pretrained_model_v2 
from tiffwrapper import make_composite 
from skimage.filters import threshold_otsu
import random
from models.UNet import UNet 
from denoising_unet import UNet as DenoisingUNet 
from diffusion_model import DDPM  
from typing import List, Dict 
from scipy.spatial import distance
from scipy.stats import mannwhitneyu, ttest_ind
from itertools import combinations
import torch 
from torch import nn 
from stedfm.decoders import get_decoder  
from stedfm.configuration import Configuration 
import argparse 
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm, trange
from wavelet import detect_spots
from skimage import measure
import json

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints-dir", type=str, default="/home-local/Frederic/baselines/SR-baselines")
parser.add_argument("--dataset", type=str, default="SynapticProteinsDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets/SynapticProteinsDataset"))
args = parser.parse_args()

def load_DM():
    # TODO: Train DM on synaptic proteins dataset
    pass 

def load_data(path: str):
    outpath = "./tmp/SynapticProteins/masks"
    os.makedirs(outpath, exist_ok=True)
    files = glob.glob(f"{path}/test_300/images/*_STED.tif")
    data = [] 
    for i, f in enumerate(files):
        image = tifffile.imread(f).squeeze()
        # image = image / 255.
        # spots = compute_mask(image)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(image, cmap="hot")
        # axs[1].imshow(spots, cmap="gray")
        # for ax in axs:
        #     ax.axis("off")
        # fig.savefig(os.path.join(outpath, f"{i}_spots.png"), dpi=1200)
        # plt.close(fig)
        fname = os.path.basename(f)
        data.append((fname, image))
    return data

def compute_mask(image: np.ndarray):
    return detect_spots(image)

def filter_files(files: List[str], model: str):
    if "Pix2Pix" in model:
        files = [f for f in files if "fake_B" in f]
    else:
        files = [f for f in files if "fakeSTED" in f]
    files = [f for f in files if "seg" not in f]
    return files

def compute_dice(ground_truth: np.ndarray, prediction: np.ndarray):
    if np.count_nonzero(ground_truth) == 0 and np.count_nonzero(prediction) == 0:
        return -1 
    elif np.count_nonzero(ground_truth) == 0 and np.count_nonzero(prediction) != 0:
        return 0.0
    # if np.count_nonzero(ground_truth) == 0:
    #     return -1.0
    else:
        intersection = np.logical_and(prediction, ground_truth)
        return (2 * intersection.sum()) / (prediction.sum() + ground_truth.sum())

def compute_statistics(data: Dict[str, List], metric: str) -> Dict:
    """
    Compute statistical differences between all unique pairs of datasets.
    
    Args:
        data: Dictionary with dataset names as keys and lists of values as data
        
    Returns:
        Dictionary containing p-values and test statistics for all pairwise comparisons
    """
    results = {
        "pairs": [],
        "mannwhitneyu_pvalue": [],
        "mannwhitneyu_statistic": [],
    }
    
    # Get all unique pairs of dataset names
    dataset_names = list(data.keys())
    pairs = list(combinations(dataset_names, 2))
    
    for name1, name2 in pairs:
        data1 = np.array(data[name1])
        data2 = np.array(data[name2])
        
        # Skip if either dataset is empty
        if len(data1) == 0 or len(data2) == 0:
            continue
            
        # Mann-Whitney U test (non-parametric)
        try:
            mw_stat, mw_pval = mannwhitneyu(data1, data2, alternative='two-sided')
        except Exception as e:
            mw_stat, mw_pval = np.nan, np.nan
        
        # Store results
        results["pairs"].append(f"{name1} vs {name2}")
        results["mannwhitneyu_statistic"].append(mw_stat)
        results["mannwhitneyu_pvalue"].append(mw_pval)


    os.makedirs(f"./tmp/SynapticProteins/stats", exist_ok=True)
    with open(f"./tmp/SynapticProteins/stats/{metric}_stats.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results


def plot_results(results: Dict, metric_key: str):
    colors = {"Pix2Pix_Proteins": "tab:green", "TAGAN_Proteins": "dodgerblue", "LALAGAN_Proteins": "#CC503E"}
    fig = plt.figure()
    ax = fig.add_subplot(111)


    stats_data = {}
    for pos, model_key in enumerate(results.keys()):
        data = results[model_key]
        stats_data[model_key] = data
        violins = ax.violinplot(data, positions=[pos], showmeans=True, showmedians=False)
        for violin in violins['bodies']:
            violin.set_facecolor(colors[model_key])
            violin.set_edgecolor('black')
        
        violins['cmeans'].set_color('black')
        violins['cmeans'].set_linewidth(0.5)
        
        violins['cbars'].set_color('black')
        violins['cbars'].set_linewidth(0.5)
        violins['cmins'].set_color('black')
        violins['cmins'].set_linewidth(0.5)
        violins['cmaxes'].set_color('black')
        violins['cmaxes'].set_linewidth(0.5)
    
    legend_elements = [Patch(facecolor=colors[key], alpha=0.5, label=key) for key in results.keys()]
    ax.legend(handles=legend_elements)
    
    if metric_key == "Dice":
        ax.set_ylim(0, 1)

    ax.set_xlabel("Model")
    ax.set_xticks([])

    ax.set_ylabel(metric_key)
    plt.savefig(f"./tmp/SynapticProteins/figures/{metric_key}.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

    stats_data = compute_statistics(stats_data, metric=metric_key)

def extract_features(image: np.ndarray, mask: np.ndarray):
    features = {"Area": [], "Eccentricity": [], "Intensity": [], "num_proteins": [], "density": []}
    mask_label, num_proteins = measure.label(mask, return_num=True)
    features["num_proteins"].append(num_proteins)
    regionprops = measure.regionprops(mask_label, intensity_image=image)

    coordinates = np.array([p.weighted_centroid for p in regionprops])
    if len(coordinates) == 0:
        features["density"].append(-1)
    elif len(coordinates) == 1:
        features["density"].append(1)
    else:
        distance_matrix = distance.cdist(coordinates, coordinates)
        distance_matrix = np.sort(distance_matrix, axis=1)
        img_density = []
        for d in range(distance_matrix.shape[0]):
            num_neighbors = np.sum(distance_matrix[d] < 50)
            img_density.append(num_neighbors)
        features["density"].append(np.mean(img_density))
    
    for prop in regionprops:
        features["Area"].append(prop.area)
        features["Eccentricity"].append(prop.eccentricity)
        features["Intensity"].append(prop.mean_intensity)

    return features

def plot_features(features: Dict, feature_key: str):
    colors = {"Ground Truth": "black", "Pix2Pix_Proteins": "tab:green", "TAGAN_Proteins": "dodgerblue", "LALAGAN_Proteins": "#CC503E"} 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for model_key in features.keys():
        model_data = np.array(features[model_key])
        x = np.sort(model_data)
        y = np.linspace(0, 1, len(x))
        ax.plot(x, y, label=model_key, color=colors[model_key])
    if feature_key == "Area":
        ax.set_xlim(0, 1000)
    ax.legend()
    ax.set_xlabel(feature_key)
    ax.set_ylabel("Cumulative frequency")
    plt.savefig(f"./tmp/SynapticProteins/figures/{feature_key}.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

        

def main():
    os.makedirs("./tmp/SynapticProteins", exist_ok=True)
    os.makedirs("./tmp/SynapticProteins/samples", exist_ok=True)
    os.makedirs("./tmp/SynapticProteins/figures", exist_ok=True)

    data = load_data(path="/home-local/Frederic/baselines/SR-baselines/results/LALAGAN_Proteins") 


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS = ["Pix2Pix_Proteins", "TAGAN_Proteins", "LALAGAN_Proteins"]

    inference_paths = [os.path.join(args.checkpoints_dir, "results", model, "test_200" if "LALAGAN" not in model else "test_400", "images") for model in MODELS]

    results = {
        "PSNR": {key: [] for key in MODELS},
        "MSE": {key: [] for key in MODELS},
        "MSSIM": {key: [] for key in MODELS},
        "Dice": {key: [] for key in MODELS},
    }
    features = {
        "Area": {key: [] for key in ["Ground Truth"] + MODELS},
        "Eccentricity": {key: [] for key in ["Ground Truth"] + MODELS},
        "Intensity": {key: [] for key in ["Ground Truth"] + MODELS},
        "num_proteins": {key: [] for key in ["Ground Truth"] + MODELS},
        "density": {key: [] for key in ["Ground Truth"] + MODELS},
    }

    for i, (fname, sted_image) in tqdm(enumerate(data), total=len(data), desc="Evaluating models..."):
        ground_truth_mask = compute_mask(sted_image)
        assert sted_image.shape == ground_truth_mask.shape == (224, 224)

        ground_truth_features = extract_features(sted_image, ground_truth_mask)
        for key in ground_truth_features.keys():
            if key == "density":
                if ground_truth_features[key] != [-1]:
                    features[key]["Ground Truth"].append(np.mean(ground_truth_features[key]))
            else:
                features[key]["Ground Truth"].extend(ground_truth_features[key])

        fig, axs = plt.subplots(2, len(MODELS) + 1)
        axs[0, 0].imshow(sted_image, cmap="hot", vmax=sted_image.max())
        axs[1, 0].imshow(ground_truth_mask, cmap="gray")

        for j, (model, path) in enumerate(zip(MODELS, inference_paths)):
            inference_files = glob.glob(f"{path}/*.tif")
            inference_files = filter_files(inference_files, model)

            temp_fname = fname.replace("_STED.tif", "")

            if "Pix2Pix" in model:
                temp_inference_files = [os.path.basename(f).replace("_fake_B.tif", "") for f in inference_files]
            else:
                temp_inference_files = [os.path.basename(f).replace("_fakeSTED.tif", "") for f in inference_files]
        

            
            found = [temp_fname == f for f in temp_inference_files]
            matching_file = inference_files[found.index(True)]
           
            pred_image = tifffile.imread(matching_file).squeeze() 
            pred_mask = compute_mask(pred_image)

            pred_features = extract_features(sted_image, pred_mask)
            for key in pred_features.keys():
                if key == "density":
                    if pred_features[key] != [-1]:
                        features[key][model].append(np.mean(pred_features[key]))
                else:
                    features[key][model].extend(pred_features[key])

            psnr = peak_signal_noise_ratio(sted_image, pred_image)
            mse = normalized_root_mse(sted_image, pred_image)

            torch_sted = torch.from_numpy(sted_image / 255.).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
            torch_pred = torch.from_numpy(pred_image / 255.).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
            mssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")(torch_sted, torch_pred).item()
            results["PSNR"][model].append(psnr)
            results["MSE"][model].append(mse)
            results["MSSIM"][model].append(mssim)

          
            
            dice = compute_dice(ground_truth_mask, pred_mask)
            if dice != -1:
                results["Dice"][model].append(dice)


            axs[0, j+1].imshow(pred_image, cmap="hot", vmax=sted_image.max())
            axs[1, j+1].imshow(pred_mask, cmap="gray")
            axs[1, j+1].set_title(f"Dice: {dice:.2f}")
            for ax in axs.ravel():
                ax.axis("off")
            axs[0, j+1].set_title(model)

        for ax in axs.ravel():
            ax.axis("off")
        axs[0, 0].set_title("STED")
        axs[0, 1].set_title("Pix2Pix")
        axs[0, 2].set_title("TAGAN")
        axs[0, 3].set_title("LALAGAN")
        if random.random() < 0.01:
            fig.savefig(f"./tmp/SynapticProteins/samples/sample_{i}.pdf", pad_inches=0.1, dpi=1200, bbox_inches='tight')
        plt.close(fig)
    
    for metric_key in results.keys():
        plot_results(results[metric_key], metric_key)

    for feature_key in features.keys():
        plot_features(features[feature_key], feature_key)

                

if __name__ == "__main__":
    main()