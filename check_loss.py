import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import os
from collections import defaultdict
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--model", type=str, default="TAGANDendriticFactin_2025-09-25")
args = parser.parse_args()

def edit_string(string: str):
    str_split = string.split(" ")
    new_str = "".join(str_split[1:])
    return new_str

def load_losses(filepath: str):
    results = {
        "G_GAN": defaultdict(list),
        "D_real": defaultdict(list),
        "D_fake": defaultdict(list),
        "S_fake": defaultdict(list)
    }

    with open(filepath, "r") as handle:
        for line in handle:
            
            epoch = line.split(" ")[1].strip().replace(",", "")
            losses = line.split(")")[-1].replace(":", "").strip().split(" ")
            
            for k in results.keys():
            
                try:
                    loss_idx = losses.index(k)
                    results[k][epoch].append(float(losses[loss_idx + 1]))
            
                except Exception as e:
                    print(e)
                    continue

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in results.keys():
        epoch_losses = [np.mean(results[k][epoch]) for epoch in results[k].keys()]
        ax.plot(epoch_losses, label=k)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    fig.savefig(os.path.join(args.checkpoints, args.model, "loss_plot.png"), dpi=1000, bbox_inches="tight")
    plt.close(fig)
            

if __name__=="__main__":
    filepath = os.path.join(args.checkpoints, args.model, "loss_log.txt")
    load_losses(filepath)