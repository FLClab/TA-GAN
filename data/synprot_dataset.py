import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
import torch
import tifffile

class SynprotDataset(BaseDataset):
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

        Returns a dictionary that contains confocal, STED, spots and image_paths
            confocal(tensor) - - a confocal image 
            STED (tensor) - - its corresponding STED image
            spots (tensor) - - the STED image wavelet segmentation
            image_paths (str) - - image paths
        """
        # read a image given a random integer index
        image_path = self.image_paths[index]
        img = tifffile.imread(image_path)
        spots561 = Image.fromarray(img[2])  
        spots640 = Image.fromarray(img[5])  
        conf561 = Image.fromarray(img[0])   
        conf640 = Image.fromarray(img[3])   
        sted561 = Image.fromarray(img[1])  
        sted640 = Image.fromarray(img[4])

        # Transform
        transform_params = get_params(self.opt, conf561.size)
        transform = get_transform(self.opt, transform_params, grayscale=True)
        transform_spots = get_transform(self.opt, transform_params, grayscale=True)

        spots561 = transform_spots(spots561)
        conf561 = transform(conf561)
        conf561 = ((conf561 - conf561.min()) / (conf561.max()-conf561.min()) - 0.5)/0.5
        sted561 = transform(sted561)
        sted561 = ((sted561 - sted561.min()) / (sted561.max()-sted561.min()) - 0.5)/0.5

        spots640 = transform_spots(spots640)
        conf640 = transform(conf640)
        conf640 = ((conf640 - conf640.min()) / (conf640.max()-conf640.min()) - 0.5)/0.5
        sted640 = transform(sted640)
        sted640 = ((sted640 - sted640.min()) / (sted640.max()-sted640.min()) - 0.5)/0.5

        conf = torch.cat((conf561, conf640), 0)
        sted = torch.cat((sted561, sted640), 0)
        spots = torch.cat((spots561, spots640), 0)

        return {'confocal': conf, 'STED': sted, 'spots': spots, 'image_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
