import torch
import numpy
from .base_model import BaseModel
from . import networks
from PIL import Image
import tifffile
import torch.multiprocessing as mp
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve as filter2
import time
mp = mp.get_context('spawn')

class Test2Model(BaseModel):
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
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.add_argument('--num_gen', type=int, default=10, help='number of synthetic STEDs generated')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['conf']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'S']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      True, opt.init_type, opt.init_gain, self.gpu_ids) # true = not opt.no_dropout
        self.netS = networks.define_S(opt.output_nc, 2, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)

    def HS(self, im1, im2, alpha, Niter):
        """
        im1: image at t=0
        im2: image at t=1
        alpha: regularization constant
        Niter: number of iteration
        """

    	#set up initial velocities
        uInitial = numpy.zeros([im1.shape[0],im1.shape[1]])
        vInitial = numpy.zeros([im1.shape[0],im1.shape[1]])

    	# Set initial value for the flow vectors
        U = uInitial
        V = vInitial

    	# Estimate derivatives
        [fx, fy, ft] = self.computeDerivatives(im1, im2)

    	# Averaging kernel
        kernel=numpy.array([[1/12, 1/6, 1/12],
                          [1/6,    0, 1/6],
                          [1/12, 1/6, 1/12]],float)

    	# Iteration to reduce error
        for _ in range(Niter):
            #%% Compute local averages of the flow vectors
            uAvg = filter2(U,kernel)
            vAvg = filter2(V,kernel)
            #%% common part of update step
            der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
            #%% iterative step
            U = uAvg - fx * der
            V = vAvg - fy * der

        return U,V

    def computeDerivatives(self, im1, im2):
    #%% build kernels for calculating derivatives
        kernelX = numpy.array([[-1, 1],
                             [-1, 1]]) * .25 #kernel for computing d/dx
        kernelY = numpy.array([[-1,-1],
                             [ 1, 1]]) * .25 #kernel for computing d/dy
        kernelT = numpy.ones((2,2))*.25

        fx = filter2(im1,kernelX) + filter2(im2,kernelX)
        fy = filter2(im1,kernelY) + filter2(im2,kernelY)

        #ft = im2 - im1
        ft = filter2(im1,kernelT) + filter2(im2,-kernelT)

        return fx,fy,ft

    def apply_HS(self, i):
        fake0 = self.fake_B_array[:,:,i].astype('float64')
        fake1 = self.fake_B_array[:,:,i+1].astype('float64')
        time_before = time.time()
        [U,V] = self.HS(fake0, fake1, 1, 100)
        return V

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.input = torch.tensor(input['INPUT']).to(self.device).unsqueeze(0)
        self.input = self.input.type(torch.FloatTensor)
        self.real_STED = torch.tensor(input['STED']).to(self.device).unsqueeze(0).unsqueeze(0)
        self.real_STED = self.real_STED.type(torch.FloatTensor)

        # input[0] (confocal) and input[2] (binary square map) should already be ok
        # input[1] is between [0, x<1], should first be normalized to 0.5 mean and std ([-1,1])
        self.input[0][1] = ( self.input[0][1] - 0.5 ) / 0.5

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.input).cpu().detach()

    def compute_std_map(self):
        time_before = time.time()
        self.fake_B_array = numpy.zeros((self.input.shape[2],self.input.shape[3],self.opt.num_gen))
        #self.seg_rings_array = numpy.zeros((self.input.shape[2],self.input.shape[3],self.opt.num_gen))
        self.seg_fibers_array = numpy.zeros((384,384,self.opt.num_gen))
        for i in range(self.opt.num_gen): # Generate N predictions
            fake_B = self.netG(self.input).cpu().detach()
            self.fake_B_array[:,:,i] = fake_B
            seg_fB = self.netS(fake_B[:,:,58:442,58:442]).cpu().detach()
            #self.seg_rings_array[:,:,i] = seg_fB[0,0,:,:] > 60
            self.seg_fibers_array[:,:,i] = seg_fB[0,1,:,:]
        # normalization
        self.real_STED = (self.real_STED / self.real_STED.max()) * 2.0 - 1.0
        self.seg_real_STED = self.netS(self.real_STED[:,:,58:442,58:442]).cpu().detach()

        print('Generating {} synthetics took {} seconds'.format(self.opt.num_gen, time.time()-time_before))
        # normalize between 0 and 1 before computing the STD, to make sure the STD is not negative
        #fake_B_array = (fake_B_array - fake_B_array.min())/(fake_B_array.max() - fake_B_array.min())
        #self.std_map = np.std(fake_B_array,axis=2)/(np.mean(fake_B_array,axis=2)+0.001)

        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(4)#mp.cpu_count())

        # Step 2: `pool.apply` the `apply_HS`
        #results = [pool.apply(apply_HS, args=(i,)) for i in range(1,50)]
        time_before = time.time()
        results = pool.map(self.apply_HS, numpy.arange(0,self.opt.num_gen-1))

        # Step 3: Don't forget to close
        pool.close()
        self.std_map = numpy.mean(results,axis=0)
        print('Computing the optical flow took {} seconds'.format(time.time()-time_before))

    def next_acquisition(self):
        # Define criteria to acquire next STED
        confidence = 0
        segmentation = 1
        if confidence:
            # Threshold segmentation map
            self.seg_fibers_array = self.seg_fibers_array > -0.5294
            seg_sum = numpy.sum(self.seg_fibers_array, axis=2)
            # Count how many pixels are segmented in [1,2,9,10]
            certain_px = (seg_sum == 1).sum() + (seg_sum == 2).sum() + (seg_sum == 9).sum() + (seg_sum == 10).sum()
            # Count how many pixels are segmented in [3,4,5,6,7,8]
            uncertain_px = (seg_sum == 3).sum() + (seg_sum == 4).sum() + (seg_sum == 5).sum() + (seg_sum == 6).sum() + (seg_sum == 7).sum() + (seg_sum == 8).sum()
            print(certain_px, uncertain_px)
            if uncertain_px > certain_px:
                print('The next image will be acquired with : STED')
                self.nextSTED = 1
            else:
                print('The next image will be acquired with : Confocal')
                self.nextSTED = 0

        if segmentation:
            seg = numpy.mean(self.seg_fibers_array, axis=2) > -0.5294
            gt = (self.seg_real_STED > -0.5294).numpy()[0,1,:,:]
            dice = numpy.sum(seg[gt==1])*2.0 / (numpy.sum(seg) + numpy.sum(gt))
            print('DC: ', dice)
            if dice < 0.3:
                print('The next image will be acquired with : STED')
                self.nextSTED = 1
            else:
                print('The next image will be acquired with : Confocal')
                self.nextSTED = 0

    def optimize_parameters(self):
        pass
