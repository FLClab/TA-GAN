from torch.utils.data import Dataset 
from typing import List, Tuple
import tifffile 
import torch 
import numpy as np

class DendriticFActinDataset(Dataset):
    def __init__(
            self,
            files: List[str],
    ) -> None:
        self.files = files 

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file = self.files[idx]
        data = tifffile.imread(file)
        confocal, sted = data[0, :, :], data[1, :, :]
        rings, fibers = data[2, :, :], data[3, :, :]
        m_conf, M_conf = np.min(confocal), np.max(confocal)
        m_sted, M_sted = np.min(sted), np.max(sted)
        confocal = (confocal - m_conf) / (M_conf - m_conf)
        sted = (sted - m_sted) / (M_sted - m_sted)
        
        confocal = torch.tensor(confocal[np.newaxis, ...], dtype=torch.float32)
        sted = torch.tensor(sted[np.newaxis, ...], dtype=torch.float32)
        rings = torch.tensor(rings[np.newaxis, ...], dtype=torch.float32)
        fibers = torch.tensor(fibers[np.newaxis, ...], dtype=torch.float32)
        ground_truth = torch.cat([rings, fibers], dim=0)
        metadata = {"min_conf": m_conf, "max_conf": M_conf, "min_sted": m_sted, "max_sted": M_sted}
        return {"confocal": confocal, "STED": sted, "seg_GTrings": rings, "seg_GTfibers": fibers, "image_paths": file}