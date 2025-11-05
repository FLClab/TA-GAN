import numpy as np 
import matplotlib.pyplot as plt 
import os 
import tifffile
import glob
import argparse 
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home-local/Frederic/Datasets/SynapticProteinsDataset/SynProt_seg/registered_test")
parser.add_argument("--outpath", type=str, default="/home-local/Frederic/Datasets/SynapticProteinsDataset/SynProt_seg_crops/registered_test")
args = parser.parse_args()

def load_data(path: str, outpath: str, crop_size: int = 224, step_size: int = 112):
    files = glob.glob(f"{path}/*.tif")
    for f in tqdm(files):
        img = tifffile.imread(f)
        _, ymax, xmax = img.shape 

        ys = np.arange(0, ymax - crop_size, step_size)
        xs = np.arange(0, xmax - crop_size, step_size)
        for y in ys:
            for x in xs:
                confocal561_crop = img[0, y:y+crop_size, x:x+crop_size] 
                sted561_crop = img[1, y:y+crop_size, x:x+crop_size] 
                # seg561_crop = img[2, y:y+crop_size, x:x+crop_size] 
                # confocal640_crop = img[3, y:y+crop_size, x:x+crop_size] 
                # sted640_crop = img[4, y:y+crop_size, x:x+crop_size] 
                # seg640_crop = img[5, y:y+crop_size, x:x+crop_size] 

             
                if confocal561_crop.shape != (224, 224):
                    continue
                else:
                    basename = f.split("/")[-1]
                    data = np.stack([confocal561_crop, sted561_crop], axis=0)
                    fname = f"{basename}_crop_{y}_{x}.tif"

                    tifffile.imwrite(os.path.join(outpath, fname), data)

def main():
    os.makedirs(args.outpath, exist_ok=True)
    load_data(path=args.path, outpath=args.outpath)

if __name__=="__main__":
    main()