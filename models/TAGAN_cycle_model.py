import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pickle
from util.util import load
import argparse
import os
import models.UNet as UNet


class TAGANCycleModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(norm='batch', netG='resnet_9blocks', netS='resnet_6blocks', dataset_mode='fixed_live', preprocess='crop_rotation', crop_size=224, batch_size=1)
        if is_train:
            parser.set_defaults(gan_mode='vanilla')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_seg', type=float, default=1, help='weight for segmentation loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_F', 'G_F', 'cycle_F', 'D_L', 'G_L', 'cycle_L', 'seg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['real_F', 'fake_L', 'rec_F', 'real_L', 'fake_F', 'rec_L', 'seg_F', 'seg_rec_F']
        else:
            self.visual_names = ['fake_L', 'seg_F']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_F', 'G_L', 'D_F', 'D_L']
        else:  # during test time, only load Gs
            self.model_names = ['G_F', 'G_L']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_F = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_L = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netS = networks.define_S(opt.input_nc, 2, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_F = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_L = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_F_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_L_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_F_pool_valid = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_L_pool_valid = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionSeg = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_F.parameters(), self.netG_L.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_F.parameters(), self.netD_L.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


        # load network from STEDActinFCNDendrite
        net_params = load('checkpoints/UNet_Dendrites', True)
        trainer_params = pickle.load(open(os.path.join('checkpoints/UNet_Dendrites', "params_trainer.pkl"), "rb"))
        network = UNet.UNet(in_channels=trainer_params["in_channels"], out_channels=trainer_params["out_channels"],
                            number_filter=trainer_params["number_filter"], depth=trainer_params["depth"],
                            size=trainer_params["size"])
        network.load_state_dict(net_params)
        if len(opt.gpu_ids) > 0:
            network = network.cuda(self.opt.gpu_ids[0])

        self.netS = network

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_F = input['F'].to(self.device)
        self.real_L = input['L'].to(self.device)
        self.image_paths = input['F_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_L = self.netG_F(self.real_F)  # G_A(A)
        self.rec_F = self.netG_L(self.fake_L)   # G_B(G_A(A))
        self.fake_F = self.netG_L(self.real_L)  # G_B(B)
        self.rec_L = self.netG_F(self.fake_F)   # G_A(G_B(B))

        # Resize real F for segmentation network
        self.real_F -= self.real_F.min()
        self.real_F /= (0.8 * (self.real_F.max() - self.real_F.min()))

        self.seg_F = self.netS(self.real_F).detach() # S_B(B)
        self.seg_F[:,0,:,:] = (self.seg_F[:,0,:,:]>0.25)
        self.seg_F[:,1,:,:] = (self.seg_F[:,1,:,:]>0.40)

        self.seg_rec_F = self.netS(self.rec_F).detach()# S_B(B)
        self.seg_rec_F[:,0,:,:] = (self.seg_rec_F[:,0,:,:]>0.25)
        self.seg_rec_F[:,1,:,:] = (self.seg_rec_F[:,1,:,:]>0.40)

        self.seg_rec_F = self.netS(self.rec_F)
        if not self.isTrain:
        #    self.seg_F = self.netS(self.real_F).detach() # S_B(B)
            self.fake_F = torch.clamp(self.fake_F - 1, -1, 1)
            self.real_F = torch.clamp(self.real_F - 1, -1, 1)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if not self.isValid:
            loss_D.backward()
        return loss_D

    def backward_D_F(self):
        """Calculate GAN loss for discriminator D_A"""
        if self.isValid:
            fake_L = self.fake_L_pool_valid.query(self.fake_L)
        else:
            fake_L = self.fake_L_pool.query(self.fake_L)
        self.loss_D_F = self.backward_D_basic(self.netD_F, self.real_L, fake_L)

    def backward_D_L(self):
        """Calculate GAN loss for discriminator D_B"""
        if self.isValid:
            fake_F = self.fake_F_pool_valid.query(self.fake_F)
        else:
            fake_F = self.fake_F_pool.query(self.fake_F)
        self.loss_D_L = self.backward_D_basic(self.netD_L, self.real_F, fake_F)    

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G_A(A))
        self.loss_G_F = self.criterionGAN(self.netD_F(self.fake_L), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_L = self.criterionGAN(self.netD_L(self.fake_F), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_F = self.criterionCycle(self.rec_F, self.real_F) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B|        
        self.loss_cycle_L = self.criterionCycle(self.rec_L, self.real_L) * lambda_B
        self.loss_seg = self.criterionSeg(self.seg_F, self.seg_rec_F) * self.opt.lambda_seg
        # combined loss and calculate gradients
        if not self.isValid:
            self.loss_G = self.loss_G_F + self.loss_G_L + self.loss_cycle_F + self.loss_cycle_L + self.loss_seg
            self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_F, self.netD_L], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_F, self.netD_L], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_F()      # calculate gradients for D_A
        self.backward_D_L()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
