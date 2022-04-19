import torch
from .base_model import BaseModel
from . import networks
from torchvision import models
import itertools

class SegmentationModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_MSE * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netS='unet_128', dataset_mode='two_masks_channels', output_nc=2)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['S']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['B', 'seg_B', 'seg_GT']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['S']
        self.netS = networks.define_S(opt.input_nc, opt.output_nc, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.criterionSEG = torch.nn.MSELoss()
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
        self.seg_rings = input['C'].to(self.device)
        self.seg_fibers = input['D'].to(self.device)
        self.seg_GT = torch.cat([self.seg_rings, self.seg_fibers], dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.seg_B = self.netS(self.B)  # S_B(B)
        if self.isTrain:
            self.loss_S = self.criterionSEG(self.seg_GT, self.seg_B)  # loss between real B segmentation and GT

    def backward_S(self):
        self.loss_S.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_S.zero_grad()        # set S's gradients to zero
        self.backward_S()                   # calculate gradients for S
        self.optimizer_S.step()             # udpate S's weights, don't if fine-tuning
