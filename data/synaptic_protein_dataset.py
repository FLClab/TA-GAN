import os 
import numpy as np
import torch 
import tifffile 
import glob
from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset 
from torchvision import transforms

class SynapticProteinDataset(Dataset):
    def __init__(
        self,
        basepath: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        task: str = "seg",
    ) -> None:
        if "test" not in split:
            path = os.path.join(basepath, f"SynProt_{task}", split)
        else:
            path = os.path.join(basepath, f"SynProt_{task}_crops", split)
        files = glob.glob(f"{path}/*.tif", recursive=True)
        self.files = list(set(files))
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_file = self.files[idx]
        data = tifffile.imread(img_file)
        if self.split == "registered_test":
            conf = torch.tensor(data[0] / 255., dtype=torch.float32).unsqueeze(0)
            sted = torch.tensor(data[1] / 255., dtype=torch.float32).unsqueeze(0)
            seg = torch.zeros_like(conf)
            # seg = torch.tensor(data[2] / data[2].max(), dtype=torch.float32).unsqueeze(0)
            return {"confocal": conf, "STED": sted, "spots": seg, "image_paths": img_file}
        else:
            conf561 = torch.tensor(data[0] / 255., dtype=torch.float32).unsqueeze(0)
            sted561 = torch.tensor(data[1] / 255., dtype=torch.float32).unsqueeze(0)
            seg561 = torch.tensor(data[2] / data[2].max(), dtype=torch.float32).unsqueeze(0)
            conf640 = torch.tensor(data[3] / 255., dtype=torch.float32).unsqueeze(0) 
            sted640 = torch.tensor(data[4] / 255., dtype=torch.float32).unsqueeze(0)
            seg640 = torch.tensor(data[5] / data[4].max(), dtype=torch.float32).unsqueeze(0)
            conf = conf561 #torch.cat([conf561, conf640], dim=0)
            sted = sted561 #torch.cat([sted561, sted640], dim=0)
            seg = seg561 #torch.cat([seg561, seg640], dim=0)

        if self.split != "test":
            conf = transforms.CenterCrop(224)(conf)
            sted = transforms.CenterCrop(224)(sted)
            seg = transforms.CenterCrop(224)(seg)

        return  {"confocal": conf, "STED": sted, "spots": seg, "image_paths": img_file}

    
