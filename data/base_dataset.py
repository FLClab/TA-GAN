"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    # crop position
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    # 0.5 probability of flipping vertically and horizontally
    flip_h = random.random() > 0.5
    flip_v = random.random() > 0.5

    # rotation angle
    angle = random.choice([0,90,180,270])

    return {'crop_pos': (x, y), 'flip_h': flip_h, 'flip_v': flip_v, 'angle': angle}


def get_transform(opt, params=None, grayscale=False, method=Image.NEAREST, convert=True, contrast=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))

    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if 'center' in opt.preprocess:
        transform_list.append(transforms.CenterCrop(opt.crop_size))

    if 'tophalf' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __crop_th(img)))

    if 'bottomhalf' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __crop_bh(img)))

    #if opt.preprocess == 'none': # changed from == 'none' to include images with shapes smaller than crop_size
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())
        if params['flip_h']:
            transform_list.append(transforms.Lambda(lambda img: __flip_h(img, params['flip_h'])))
        if params['flip_v']:
            transform_list.append(transforms.Lambda(lambda img: __flip_v(img, params['flip_v'])))

    if 'rotation' in opt.preprocess:
        if params is None:
            angle = random.choice([0,90,180,270])
            transform_list.append(transforms.RandomRotation((angle,angle)))
        if params['angle']:
            transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['angle'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.NEAREST):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    # Absolutely inefficient way to make sure everything is 200x200:
    #if (oh != 200) or (ow != 200): # un-comment for training
    #    delta_w = 200 - ow
    #    delta_h = 200 - oh
    #    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    #    return ImageOps.expand(img, padding)
    #else:
    #    return img
    if (h == oh) and (w == ow):
        return img
    else:
        __print_size_warning(ow, oh, w, h)
        return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __crop_th(img):
    ow, oh = img.size
    x1 = y1 = 0
    return img.crop((x1, y1, x1 + ow, y1 + oh//2))

def __crop_bh(img):
    ow, oh = img.size
    x1 = 0
    y1 = oh // 2
    return img.crop((x1, y1, x1 + ow, y1 + oh//2))

def __flip_h(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __flip_v(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def __rotate(img, angle):
    return img.rotate(angle)

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
