import torch
from .base_model import BaseModel
from . import networks
from torchvision import models, transforms
import itertools
import pickle
from util.util import load
import numpy
import tifffile
import os
import models.UNet as UNet

class TAGANLiveModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='resnet_9blocks', netS='resnet_6blocks', dataset_mode='aligned', input_nc=1, output_nc=1)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_seg', type=float, default=1, help='weight for seg loss')
            parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for GAN loss')
        else:
            parser.add_argument('--num_gens', type=int, default=1, help='number of generations (test only)')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        torch.cuda.empty_cache()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'S_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.visual_names = ['input', 'real_B', 'seg_rB', 'seg_fB', 'fake_B', 'S', 'seg_GT', 'decision_map']
        self.visual_names = ['input', 'STED', 'fakeSTED']
        if not self.isTrain:
            self.isValid = False

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (generator, discriminator)
        self.netG = networks.define_G(3, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) # not opt.no_dropout

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc+opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSEG = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netG2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # load network from STEDActinFCNDendrite
        net_params = load(os.path.join('checkpoints','UNet_Dendrites'), True)
        trainer_params = pickle.load(open(os.path.join('checkpoints','UNet_Dendrites', "params_trainer.pkl"), "rb"))
        network = UNet.UNet(in_channels=trainer_params["in_channels"], out_channels=trainer_params["out_channels"],
                            number_filter=trainer_params["number_filter"], depth=trainer_params["depth"],
                            size=trainer_params["size"])
        network.load_state_dict(net_params)
        self.netS = network.cuda(self.opt.gpu_ids[0])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.confocal = input['confocal'].to(self.device) 
        self.STED = input['STED'].to(self.device) 
        self.decision_map = input['decision_map'].to(self.device)
        # Concatenate confocal and STED_map to two channels input
        self.S = self.STED * self.decision_map
        self.S[self.decision_map==0] = -1
        self.decision_map[self.decision_map==0] = -1
        self.input = torch.cat((self.confocal, self.S, self.decision_map), 1)
        self.image_paths = input['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.fakeSTED = self.netG(self.input)
            self.seg_fakeSTED = self.netS(self.fakeSTED)
            self.seg_STED = self.netS(self.STED).detach()
            # Threshold the segmentation map
            self.seg_STED[:,0,:,:] = self.seg_STED[:,0,:,:]>(0.07*2-1)
            self.seg_STED[:,1,:,:] = self.seg_STED[:,1,:,:]>(0.04*2-1)

        else: # Testing
            self.fakeSTED = torch.zeros((self.opt.num_gens,1,self.input.shape[2],self.input.shape[3]))
            self.seg_STED = self.netS(self.STED)
            for i in range(self.opt.num_gens):
                fakeSTED = self.netG(self.input)
                self.fakeSTED[i,:,:,:] = fakeSTED
            print(self.fakeSTED.shape)
            
            # Segmentation
            if 'seg_fakeSTED0' in self.visual_names:
                self.seg_fakeSTED0 = torch.zeros((self.opt.num_gens,1,self.input.shape[2],self.input.shape[3]))
                self.seg_fakeSTED1 = torch.zeros((self.opt.num_gens,1,self.input.shape[2],self.input.shape[3]))
                for i in range(self.opt.num_gens):
                    seg_fakeSTED = self.netS(fake_STED)
                    self.seg_fakeSTED0[i,:,:,:] = seg_fakeSTED[:,0,:,:]
                    self.seg_fakeSTED1[i,:,:,:] = seg_fakeSTED[:,1,:,:]
                self.seg_fakeSTED0 = torch.mean(self.seg_fakeSTED0, dim=0)
                self.seg_fakeSTED1 = torch.mean(self.seg_fakeSTED1, dim=0)        


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_confSTED = torch.cat((self.confocal, self.fakeSTED), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_confSTED.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_confSTED = torch.cat((self.confocal, self.STED), 1)
        pred_real = self.netD(real_confSTED)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Fake; stop backprop to the generator by detaching fake_B
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and MSE loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_confSTED = torch.cat((self.confocal, self.STED), 1)
        pred_fake = self.netD(fake_confSTED)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # Segmentation loss S(G(A)) = S(B)
        self.loss_S_fake = self.criterionSEG(self.seg_fakeSTED, self.seg_STED) * self.opt.lambda_seg

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_S_fake
        self.loss_G.backward()

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
