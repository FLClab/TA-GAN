import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch import nn 
from torch.utils.data import Dataset

class SynapticProteinDataset(Dataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)