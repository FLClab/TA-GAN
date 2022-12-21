import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
import torch
import tifffile

class SAureusDataset(BaseDataset):
    """A dataset class for the S Aureus dataset, available at https://zenodo.org/record/5550933#.Y2F3-dLMJH5.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_images = os.path.join(opt.dataroot, opt.phase, 'SIM')  # get the image directory
        self.image_paths = sorted(make_dataset(self.dir_images, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc
        self.output_nc = self.opt.input_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            confocal(tensor) - - a confocal image 
            STED (tensor) - - its corresponding STED image
            image_paths (str) - - image paths
        """
        # read a image given a random integer index
        image_path = self.image_paths[index]
        SIM   = tifffile.imread(image_path)
        WF    = tifffile.imread(image_path.replace('SIM', 'BF'))
        mask  = tifffile.imread(image_path.replace('SIM', 'masks')).astype('uint8')

        # normalize between 0-1
        SIM = (SIM-SIM.min())/(SIM.max()-SIM.min())
        WF = (WF-WF.min())/(WF.max()-WF.min())
        if mask.max() > 0:
            mask = (mask-mask.min())/(mask.max()-mask.min())

        # convert to PIL Image
        SIM = Image.fromarray(SIM)
        WF = Image.fromarray(WF)
        mask = Image.fromarray(mask)

        # define and apply transformations
        transform_params = get_params(self.opt, SIM.size)
        image_transform = get_transform(self.opt, transform_params, grayscale=False)

        SIM = image_transform(SIM)
        WF = image_transform(WF)
        mask = image_transform(mask)

        return {'confocal': WF, 'STED': SIM, 'seg_GT': mask, 'image_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
