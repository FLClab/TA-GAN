from torch.utils.data import Dataset 
from typing import List, Tuple, Optional, Callable
import tifffile 
import torch 
import numpy as np

class DendriticFActinDataset(Dataset):
    def __init__(
            self,
            files: List[str],
            transform: Optional[Callable] = None,
    ) -> None:
        self.files = files 
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file = self.files[idx]
        data = tifffile.imread(file)
        confocal, sted = data[0, :, :], data[1, :, :]
        rings, fibers = data[2, :, :], data[3, :, :]
        # m_conf, M_conf = np.min(confocal), np.max(confocal)
        # m_sted, M_sted = np.min(sted), np.max(sted)
        # confocal = (confocal - m_conf) / (M_conf - m_conf)
        # sted = (sted - m_sted) / (M_sted - m_sted)
        confocal = confocal / 255.0
        sted = sted / 255.0
        
        confocal = torch.tensor(confocal[np.newaxis, ...], dtype=torch.float32)
        sted = torch.tensor(sted[np.newaxis, ...], dtype=torch.float32)
        cat = torch.cat([confocal, sted], dim=0)
        cat = self.transform(cat) if self.transform is not None else cat
        confocal, sted = cat[0:1], cat[1:2]
        rings = torch.tensor(rings[np.newaxis, ...], dtype=torch.float32)
        fibers = torch.tensor(fibers[np.newaxis, ...], dtype=torch.float32)
        ground_truth = torch.cat([rings, fibers], dim=0)
        return {"confocal": confocal, "STED": sted, "seg_GTrings": rings, "seg_GTfibers": fibers, "image_paths": file}