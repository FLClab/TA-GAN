import numpy as np 
import matplotlib.pyplot as plt
import tifffile 
import os 
import glob 
from stedfm.DEFAULTS import BASE_PATH

if __name__=="__main__":
    os.makedirs("./tmp/AxonalRings", exist_ok=True)
    files = glob.glob(f"{BASE_PATH}/Datasets/AxonalRingsDataset/test/*.tif")
    print(f"[---] Found {len(files)} training files [---]")
    
    
    for i, file in enumerate(files): 
        image = tifffile.imread(file) 
        first, second, third = image[0] / 255.0, image[1] / 255.0, image[2] / 255.0
        print(first.shape, second.shape, third.shape)
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(first, cmap='hot')
        # axs[1].imshow(second, cmap='hot')
        # axs[2].imshow(third, cmap='hot')
        # for ax in axs:
        #     ax.axis("off")
        # fig.savefig(f"./tmp/AxonalRings/dummy_{i}.png", dpi=800, bbox_inches='tight', pad_inches=0)
        # plt.close()
        if i > 10:
            break