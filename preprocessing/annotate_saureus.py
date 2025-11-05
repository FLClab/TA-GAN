import numpy as np
import matplotlib.pyplot as plt 
import tifffile  
from skimage.filters import sobel 
import tarfile 
import argparse 
from scipy.ndimage import gaussian_filter
import glob 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", type=str, default="/home-local/Frederic/Datasets/SaureusDataset")
args = parser.parse_args()

def minmax_normalize(image: np.ndarray):
    m, M = image.min(), image.max()
    return (image - m) / (M - m)

def binarize_mask(mask: np.ndarray):
    return mask > 0

def load_data(path: str, split: str = "train"):
    if split == "train":
        brightfield_path = f"{path}/brightfield_dataset/{split}/full_images/brightfield/*.tif"
        mask_path = f"{path}/brightfield_dataset/{split}/full_images/masks/*.tif"
        fluo_path = f"{path}/fluorescence_dataset/{split}/full_images/fluorescence/*.tif"

    else:
        brightfield_path = f"{path}/brightfield_dataset/{split}/brightfield/*.tif"
        mask_path = f"{path}/brightfield_dataset/{split}/masks/*.tif"
        fluo_path = f"{path}/fluorescence_dataset/{split}/fluorescence/*.tif"


    brightfield_files = glob.glob(brightfield_path)
    mask_files = glob.glob(mask_path)
    fluo_files = glob.glob(fluo_path)

    print(f"\tNumber of brightfield files: {len(brightfield_files)}")
    print(f"\tNumber of mask files: {len(mask_files)}")
    print(f"\tNumber of fluorescence files: {len(fluo_files)}")
    
    paired_files = []
    for f in brightfield_files:
        brightfield_basename = os.path.basename(f)
        
        found = [brightfield_basename == os.path.basename(m) for m in mask_files]
        paired_mask = mask_files[found.index(True)]

        found = [brightfield_basename == os.path.basename(m) for m in fluo_files]
        paired_fluo = fluo_files[found.index(True)]
    

        paired_files.append((f, paired_fluo, paired_mask))
    return paired_files

def process_data(data, i):
    brightfield_file, fluo_file, mask_file = data 
    brightfield_image = tifffile.imread(brightfield_file) 
    fluo_image = tifffile.imread(fluo_file)
    mask_image = tifffile.imread(mask_file)
    mask_image = binarize_mask(mask_image)
    brightfield_image = minmax_normalize(brightfield_image)
    fluo_image = minmax_normalize(fluo_image)

    sim_sobel = sobel(fluo_image)
    sim_filtered = gaussian_filter(sim_sobel, sigma=1)
    sim_threshold = np.max(sim_filtered) * 0.20
    sim_final = (sim_filtered > sim_threshold).astype(np.uint8)

    mask_sobel = sobel(mask_image)
    mask_filtered = gaussian_filter(mask_sobel, sigma=0.1)
    mask_final = (mask_filtered > 0).astype(np.uint8)

    hr_annot = sim_final - mask_final


    fig, axs = plt.subplots(1, 6, figsize=(15, 5))
    axs[0].imshow(brightfield_image, cmap="gray", vmax=1)
    axs[0].set_title("Brightfield")
    axs[1].imshow(fluo_image, cmap="gray", vmax=1)
    axs[1].set_title("Fluorescence")
    axs[2].imshow(mask_image, cmap="gray", vmax=1)
    axs[2].set_title("Mask")
    axs[3].imshow(sim_final, cmap="gray", vmax=1)
    axs[3].set_title("division boundary")
    axs[4].imshow(mask_final, cmap="gray", vmax=1)
    axs[4].set_title("edges")
    axs[5].imshow(hr_annot, cmap="gray", vmax=1)
    axs[5].set_title("HR annotation")
    for ax in axs:
        ax.axis("off")
    fig.savefig(f"./saureus_{i}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def main():

    print("*********** Train set ***********")
    paired_files = load_data(args.input_path) 
    for i, p in enumerate(paired_files):
        assert len(p) == 3
        process_data(p, i)

    print("*********** Test set ***********")
    paired_files = load_data(args.input_path, split="test")
    for p in paired_files:
        assert len(p) == 3
    

if __name__=="__main__":
    main()