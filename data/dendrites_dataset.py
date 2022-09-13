import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
import torch
import tifffile

class DendritesDataset(BaseDataset):
    """A dataset class for paired image dataset with segmentation mask fo one structure.

    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_images = os.path.join(opt.dataroot, opt.phase)  # get the image directory
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

        image = tifffile.imread(image_path)
        # split image into confocal and STED
        confocal = Image.fromarray(image[0])
        STED = Image.fromarray(image[1])
        seg_GTrings = Image.fromarray(image[2])
        seg_GTfibers = Image.fromarray(image[3])

        transform_params = get_params(self.opt, confocal.size)
        image_transform = get_transform(self.opt, transform_params, grayscale=True)

        confocal = image_transform(confocal)
        STED = image_transform(STED)
        seg_GTrings = image_transform(seg_GTrings)
        seg_GTfibers = image_transform(seg_GTfibers)

        return {'confocal': confocal, 'STED': STED, 'seg_GTrings': seg_GTrings, 'seg_GTfibers': seg_GTfibers, 'image_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
