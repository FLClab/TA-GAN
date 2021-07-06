import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
import torch
import random
import tifffile
import skimage
from skimage import measure
from torchvision import transforms

class LiveSTEDPairedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = tifffile.imread(AB_path)
        if AB.min() == 32768:
            AB = AB - 32768
        # split AB image into A and B
        h, w = AB.shape
        w2 = int(w / 2)
        A = AB[:, :w2]
        B = AB[:, w2:]
        A = Image.fromarray(A)
        B = Image.fromarray(B)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)

        A = A_transform(A)
        B = B_transform(B)

        # Add channel consisting of real STED crops
        # First, simplify the confocal to a map of super-pixels
        px = random.choice([100]) # pixel size
        sup_px_map = skimage.measure.block_reduce(A.numpy()[0], (px,px), numpy.mean)
        sorted_list = numpy.sort(sup_px_map.ravel())
        n = random.choice(range(0,int(0.5*len(sorted_list)+1)))#int(random.uniform(0,0.5) * len(sorted_list))
        sup_px_map = sup_px_map > sorted_list[-(n + 1)]
        sup_px_map = Image.fromarray(sup_px_map).resize((A.shape[2], A.shape[1])) # 0-1
        tf = transforms.ToTensor()
        sup_px_map = tf(sup_px_map)# * numpy.random.randint(2)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'decision_map': sup_px_map}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
