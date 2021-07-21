"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import csv
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    opt_val = TrainOptions().parse()           # create options for your validation dataset
    opt_val.phase = 'valid'    # specify where your validation images are saved
    opt_val.preprocess = 'center'    # you don't want data-augmentation in validation, unless you're using U-Net, then you might need to crop! If so, just remove this line.
    opt_val.crop_size = 224
    opt_val.no_flip = True
    opt_val.serial_batches = True  # with this option, it's always the same validatoin image that is saved during training, which helps with seeing the evolution of the performance
    opt_val.batch_size = 1
    dataval = create_dataset(opt_val)  # create the validation dataset
    print('The number of training images = %d' % dataset_size)
    print('The number of validation images = %d' % len(dataval))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    f = 'checkpoints/'+str(opt.name)+'/loss.csv'
    with open(f,'a') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch'] + model.loss_names * 2)

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        D_A, G_A, cycle_A, D_B, G_B, cycle_B, seg, vD_A, vG_A, vcycle_A, vD_B, vG_B, vcycle_B, vseg = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.isValid = False
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()           # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            D_A.append(losses['D_F'])
            D_B.append(losses['D_L'])
            G_A.append(losses['G_F'])
            G_B.append(losses['G_L'])
            cycle_A.append(losses['cycle_F'])
            cycle_B.append(losses['cycle_L'])
            seg.append(losses['seg'])

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        for i, data in enumerate(dataval):
            # Validation
            model.set_input(data)
            with torch.no_grad():
                model.isValid = True
                model.forward()
                model.compute_visuals()
                losses = model.get_current_losses()
            vD_A.append(losses['D_F'])
            vD_B.append(losses['D_L'])
            vG_A.append(losses['G_F'])
            vG_B.append(losses['G_L'])
            vcycle_A.append(losses['cycle_F'])
            vcycle_B.append(losses['cycle_L'])
            vseg.append(losses['seg'])

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        row = [epoch, sum(D_A)/len(D_A), sum(G_A)/len(G_A), sum(cycle_A)/len(cycle_A), sum(D_B)/len(D_B),
                      sum(G_B)/len(G_B), sum(cycle_B)/len(cycle_B), sum(seg)/len(seg),
                      sum(vD_A)/len(vD_A), sum(vG_A)/len(vG_A), sum(vcycle_A)/len(vcycle_A), sum(vD_B)/len(vD_B),
                      sum(vG_B)/len(vG_B), sum(vcycle_B)/len(vcycle_B), sum(vseg)/len(vseg)]
        with open(f,'a') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        model.compute_visuals()
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result=True)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
