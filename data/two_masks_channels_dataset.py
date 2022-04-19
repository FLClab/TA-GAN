import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
from scipy.signal import convolve2d as conv2
import tifffile

class TwoMasksChannelsDataset(BaseDataset):
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
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ABC_paths = sorted(make_dataset(self.dir_ABC, opt.max_dataset_size))  # get image paths
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
        # read an image given a random integer index
        ABC_path = self.ABC_paths[index]
        ABC = tifffile.imread(ABC_path)
        # split AB image into A and B
        c, h, w = ABC.shape
        A = Image.fromarray(ABC[0])
        B = Image.fromarray(ABC[0])
        C = Image.fromarray(ABC[1])
        D = Image.fromarray(ABC[2])

        # apply the same transform to A, B, C and D
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)
        C_transform = get_transform(self.opt, transform_params, grayscale=True)
        D_transform = get_transform(self.opt, transform_params, grayscale=True)

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        D = C_transform(D)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'A_paths': ABC_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
