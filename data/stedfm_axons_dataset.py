import numpy as np
import tifffile 
import torch
import os 
import glob 
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable

class AxonalRingsDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file = self.files[idx]
        data = tifffile.imread(file)
        confocal, sted, rings = data[0] / 255.0, data[1] / 255.0, data[2] / 255.0
        confocal = torch.tensor(confocal[np.newaxis, ...], dtype=torch.float32)
        sted = torch.tensor(sted[np.newaxis, ...], dtype=torch.float32)
        rings = torch.tensor(rings[np.newaxis, ...], dtype=torch.float32)
        cat = torch.cat([confocal, sted, rings], dim=0)
        cat = self.transform(cat) if self.transform is not None else cat
        confocal, sted, rings = cat[0:1], cat[1:2], cat[2:3]
        return {"confocal": confocal, "STED": sted, "seg_GT": rings, "image_paths": file}
        