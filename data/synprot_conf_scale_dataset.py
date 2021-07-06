import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
from scipy.signal import convolve2d as conv2
import tifffile
import torch

class SynprotConfScaleDataset(BaseDataset):
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

    def whitening(self, im):
        """Normalize image between 0 and 1"""
        im = im.astype('float16')
        im = ((im - im.min())/im.max() * 2**16).astype('float64')
        return Image.fromarray(im/2**8)

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

        # apply the same transform to A, B, C and D
        conf561 = Image.fromarray(ABC[1,:,:]).convert('F') ####
        conf640 = Image.fromarray(ABC[0,:,:]).convert('F')

        conf561 = conf561.resize((conf561.width * 4, conf561.height * 4), resample=Image.NEAREST)
        conf640 = conf640.resize((conf640.width * 4, conf640.height * 4), resample=Image.NEAREST)

        transform_params = get_params(self.opt, conf561.size)
        transform = get_transform(self.opt, transform_params, grayscale=False)
        conf561 = transform(conf561)
        conf561 = ((conf561 - conf561.min()) / (conf561.max()-conf561.min()) - 0.5)/0.5
        conf640 = transform(conf640)
        conf640 = ((conf640 - conf640.min()) / (conf640.max()-conf640.min()) - 0.5)/0.5

        return {'conf561': conf561, 'conf640': conf640, 'A_paths': ABC_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
