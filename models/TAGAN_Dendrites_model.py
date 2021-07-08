import torch
from .base_model import BaseModel
from . import networks
from torchvision import models
import itertools

class TAGANDendritesModel(BaseModel):
    """ This class implements the TA-GAN model for confocal to STED resolution enhancement for the dendritic F-actin rings and fibers dataset.

    Original code taken from pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        
        parser.set_defaults(norm='batch', netG='resnet_9blocks', netS='resnet_6blocks', dataset_mode='two_masks')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla', niter=500, niter_decay=0, batch_size=32, preprocess='crop_rotation', crop_size=128)
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
            parser.add_argument('--lambda_seg', type=float, default=1.0, help='weight for seg loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'S_real', 'S_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['confocal', 'STED', 'fakeSTED', 'seg_STED', 'seg_fakeSTED', 'seg_GT']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'S', 'D']
        else:  # during test time, only load G and S
            self.model_names = ['G', 'S']

        # define networks (generator, discriminator and segmentation network)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netS = networks.define_S(opt.input_nc, 2, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSEG = torch.nn.MSELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.confocal = input['confocal'].to(self.device)
        self.STED = input['STED'].to(self.device)
        self.seg_GTrings = input['seg_GTrings'].to(self.device)
        self.seg_GTfibers = input['seg_GTfibers'].to(self.device)
        self.seg_GT = torch.cat([self.seg_GTrings, self.seg_GTfibers], dim=1)
        self.image_paths = input['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fakeSTED = self.netG(self.confocal)
        self.seg_STED = self.netS(self.STED) 
        self.seg_fakeSTED = self.netS(self.fakeSTED)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_confSTED = torch.cat((self.confocal, self.fakeSTED), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_confSTED.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Real
        real_confSTED = torch.cat((self.confocal, self.STED), 1)
        pred_real = self.netD(real_confSTED)
        self.loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and MSE loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_confSTED = torch.cat((self.confocal, self.fakeSTED), 1)
        pred_fake = self.netD(fake_confSTED)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # Segmentation loss
        self.loss_S_fake = self.criterionSEG(self.seg_fakeSTED, self.seg_GT) * self.opt.lambda_seg # loss between fake B segmentation and GT

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_S_fake
        self.loss_G.backward()

    def backward_S(self):
        self.loss_S_real = self.criterionSEG(self.seg_STED, self.seg_GT) * self.opt.lambda_seg # loss between real B segmentation and GT
        self.loss_S_real.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G and S
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # udpate G's weights
        # update S
        self.optimizer_S.zero_grad()        # set S's gradients to zero
        self.backward_S()                   # calculate gradients for S
        self.optimizer_S.step()             # udpate S's weights
