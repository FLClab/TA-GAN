import tifffile 
import numpy as np 
import glob 
from data.synaptic_protein_dataset import SynapticProteinDataset 
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
from stedfm.configuration import Configuration 
import argparse 
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm, trange
from wavelet import detect_spots

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints-dir", type=str, default="/home-local/Frederic/baselines/SR-baselines")
parser.add_argument("--dataset", type=str, default="SynapticProteinsDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets/SynapticProteinsDataset"))
args = parser.parse_args()

def load_DM():
    # TODO: Train DM on synaptic proteins dataset
    pass 

def main():
    pass 

if __name__ == "__main__":
    main()