import os 
import numpy 
import torch 
import tifffile 
from torch.utils.data import Dataset 

class SynapticProteinDataset(Dataset):
    def __init__(
        self,
        basepath: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        task: str = "seg",
    ) -> None:
        path = os.path.join(basepath, f"SynProt_{task}", split)
        files = glob.glob(f"{path}/*.tif", recursive=True)
        self.files = list(set(files))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_file = self.files[idx]
        data = tifffile.imread(img_file)
        conf561 = data[0]
        sted561 = data[1]
        seg561 = data[2]
        conf640 = data[3]
        sted640 = data[4] 
        seg640 = data[5]

        print(conf561.min(), conf561.max(), conf561.shape)
        print(sted561.min(), sted561.max(), sted561.shape)
        print(conf640.min(), conf640.max(), conf640.shape)
        print(sted640.min(), sted640.max(), sted640.shape)
        exit()

if __name__=="__main__":
    dataset = SynapticProteinDataset(
        basepath="/home/f/frbea320/links/scratch/Datasets/LargeProteinModels/SyntheticAnomalies",
        split="train",
        transform=None,
        task="seg"
    )
    print(dataset[42])