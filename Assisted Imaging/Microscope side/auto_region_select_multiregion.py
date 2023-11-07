import os
import functools
import warnings
import time
import csv
import json

#import matplotlib; matplotlib.use("TkAgg")

import numpy
import skimage.io
import yaml
#from abberior import microscope
import user
import utils
from datetime import date

import util
import torch
from PIL import Image
import skimage.measure
import skimage.filters
from torchvision import transforms
import tifffile

import requests
from skimage.filters import threshold_triangle, threshold_otsu, gaussian

name = 'Auto_Select_MultiRegion'
id=input('Enter sample region identifier: ')

px = 100  # pixel size (min. 50)
ROI = 300 # center region of interest where no STED will be acquired (excpet full STED)
width, height = 500.0*20*1e-9, 500.0*20*1e-9
repetitions = 15
tlim = 60

config_STED = microscope.get_config("Setting STED configuration.")
config_confocal = microscope.get_config("Setting Confocal configuration.")

regions_selected = user.get_regions()

def run(config_window, t, regions, output, imsavepath = None,rep=1):
    #  Show overview image to user and ask for rectangle selections
    overview = '640'
    width, height = 400.0*20*1e-9, 400.0*20*1e-9

    # For each rectangle selected by user, set imaging parameters, acquire image and save configuration and images
    for (x, y) in regions:

        # This will output a .csv file for every region that were imaged
        # parameters x,y are the imaging region offset center
        # width, height are the imaging size
        with open(os.path.join(imsavepath, "imaging_parameters_ROI{}".format(t)), "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([x, y, width, height])
        t += 1
        print('regions saved to csv')

        # Set microscope configurations to image selected region
        microscope.set_offsets(config_window, x, y)
        microscope.set_imagesize(config_window, width, height)

        # Set microscope configurations to set RESCue parameters
        print('microscope parameters set')

        # This will output a .csv file for every region that were imaged
        # parameters x,y are the imaging region offset center
        # width, height are the imaging size
        with open(os.path.join(imsavepath, "imaging_parameters_ROI{}.csv".format(t)), "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([x, y, width, height])
            #!!!print('regions saved to csv')
        # Acquire image and save file to .msr and images to .tiff

        #!!!print('Acquiring image')
        timepre = time.time()
        stacks, _ = microscope.acquire_saveasmsr(config_window, os.path.join(imsavepath, "_ROI{}_{}.msr".format(t,rep)))
        print('Acquiring the image took {} seconds'.format(time.time()-timepre))
        #time.sleep(120)
        #stacks, _ = microscope.acquire_saveasmsr(config_confocal, os.path.join(imsavepath, "CONF_ROI{}.msr".format(t)))
        if t==1:
            all_stacks = stacks
        else:
            all_stacks = numpy.concatenate([all_stacks, stacks], 1)

        with open(os.path.join(imsavepath, "imspector_config_window_ROI{}".format(t)), "w") as f:
            config = config_window.parameters("")
            yaml.dump(config, f)
    return t, all_stacks, x, y

def run_STED(config_window, rect_regions_offset, regions, t, output, imsavepath = None,rep=1):
    #  Show overview image to user and ask for rectangle selections
    overview = '640'
    # For each rectangle selected by user, set imaging parameters, acquire image and save configuration and images
    for (x, y), (width, height) in zip(rect_regions_offset, regions):

        # This will output a .csv file for every region that were imaged
        # parameters x,y are the imaging region offset center
        # width, height are the imaging size
        with open(os.path.join(imsavepath, "imaging_parameters_ROI{}".format(t)), "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([x, y, width, height])
        #t += 1
        #!!!print('regions saved to csv')

        # Set microscope configurations to image selected region
        microscope.set_offsets(config_window, x, y)
        microscope.set_imagesize(config_window, width, height)

        # Set microscope configurations to set RESCue parameters
        print('microscope parameters set')

        # This will output a .csv file for every region that were imaged
        # parameters x,y are the imaging region offset center
        # width, height are the imaging size
        with open(os.path.join(imsavepath, "imaging_parameters_ROI{}.csv".format(t)), "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([x, y, width, height])
            #!!!print('regions saved to csv')
        # Acquire image and save file to .msr and images to .tiff

        print('Acquiring  image')
        all_stacks, _ = microscope.acquire_saveasmsr(config_window, os.path.join(imsavepath, "_ROI{}_{}.msr".format(t,rep)))

        with open(os.path.join(imsavepath, "imspector_config_window_ROI{}".format(t)), "w") as f:
            config = config_window.parameters("")
            yaml.dump(config, f)
    return t, stacks, all_stacks

today = date.today()
folder = os.path.join("C:", os.sep, "Users", "abberior", "Desktop","DATA","Catherine",
                      "{}-{:02d}-{:02d}".format(today.year, today.month, today.day))
output = os.path.join(folder, name+id)
imsavepath = os.path.join(output, "STEDimages_final")
confimsavepath = os.path.join(output, "Confocalimages")
stedimsavepath = os.path.join(output, "STEDimages")
os.makedirs(imsavepath, exist_ok=True)
os.makedirs(confimsavepath, exist_ok=True)
os.makedirs(stedimsavepath, exist_ok=True)

class VirtualNet(object):
    """This class implements a remote network
    :param address (str): Address of the network.
    :param port (int): Port of the network (default: 5000)
    """

    def __init__(self, img, sted, address='172.16.1.170', port=5000): # 172.16.1.170/166
        self.img = img
        self.sted = sted
        self.address = address
        self.port = port
        self.url = 'http://{}:{}'.format(self.address, self.port)
        print('Running on server '+self.url)

    def predict(self):
        """Generate a fake STED from confocal input
        """
        tosend = json.dumps({'INPUT':self.img.tolist(), 'STED':self.sted.tolist()})
        #tosend = json.dumps({'INPUT':self.img.tolist()})
        #tosend = json.dumps({'STED':self.sted.tolist()})
        r = requests.post(self.url, data=tosend)
        data = json.loads(r.text)
        return numpy.array(data['std_map']), numpy.array(data['synthetic_STED']), numpy.array(data['nextSTED']), numpy.array(data['seg_real_STED']), numpy.array(data['seg_fibers'])

def update(val):
    thresh = sap.val
    sup_px_map_thresh = (sup_px_map > thresh).astype(int)
    im.set_data(sup_px_map_thresh)
    im.axes.figure.canvas.draw()

# acquire initial STED
_, full_STED_stacks, _, _ =  run(config_STED, 0, regions_selected, output, confimsavepath, rep=0)
tifffile.imsave(os.path.join(output,'initialSTED.tif'), numpy.array(full_STED_stacks).astype('uint16'))
time.sleep(1)

# Acquire initial Confocal
t, confocal_stacks, _, _ = run(config_confocal, 0, regions_selected, output, confimsavepath, rep=0)
tifffile.imsave(os.path.join(output,'initialconfocal.tif'), numpy.array(confocal_stacks).astype('uint16'))
time.sleep(1)

# get ROI map
ROI_map = numpy.zeros(confocal_stacks[0][0].shape)
ROI_map[int((ROI_map.shape[0]-ROI)/2):int((ROI_map.shape[0]+ROI)/2), int((ROI_map.shape[1]-ROI)/2):int((ROI_map.shape[1]+ROI)/2)] = 1
tifffile.imsave(os.path.join(output,'ROI.tif'), ROI_map.astype('uint8'))

# confocal is first passed through the network on its own to decide first region to acquire
# define decision maps (on first pass, everything should be 0 or -1)
sup_px_map_STED = numpy.zeros((len(regions_selected), confocal_stacks[0].shape[1], confocal_stacks[0].shape[2])) - 1 # -1 everywhere except STED region

std_map_allROI = numpy.zeros((len(regions_selected), confocal_stacks[0].shape[1], confocal_stacks[0].shape[2]))
synthetic_allROI = numpy.zeros((len(regions_selected), confocal_stacks[0].shape[1], confocal_stacks[0].shape[2]))
sup_px_map_allROI = numpy.zeros((len(regions_selected), confocal_stacks[0].shape[1], confocal_stacks[0].shape[2]))

# Uncomment for selection of a region, otherwise the top right corner is selected for the first loop
if 0 : #for ROI_idx in range(len(regions_selected)):
    confocal = Image.fromarray(confocal_stacks[0][ROI_idx])
    tf = transforms.ToTensor()
    confocal = tf(confocal)

    # normalize confocal to fit with training (gives better results when spread over range (-1,1))
    confocal = confocal.double()
    confocal = (confocal/confocal.max() - 0.5) / 0.5

    # concatenate confocal with decision maps and pass to network
    data = torch.cat((confocal, torch.from_numpy(sup_px_map_STED[ROI_idx]).unsqueeze(0), torch.from_numpy(sup_px_map_STED[ROI_idx]).unsqueeze(0)), 0)
    timepre = time.time()
    model = VirtualNet(data, full_STED_stacks[0][ROI_idx]) # Here sup_px_map is between -1 and 1
    next_STED, std_map, synthetic, seg_real, seg_fake = model.predict()
    std_map_allROI[ROI_idx] = std_map * (1-ROI_map)
    synthetic_allROI[ROI_idx] = synthetic
    print('Computing time on server: {}'.format(time.time() - timepre))

# We now enter the loop. All decision maps are empty.
for r in range(1,repetitions+1):
    # acquire new confocal
    _, confocal_stacks, _, _ = run(config_confocal, 0, regions_selected, output, confimsavepath, rep=r)
    tifffile.imsave(os.path.join(output,'confocal{}.tif'.format(r)), numpy.array(confocal_stacks).astype('uint16'))
    ROI_idx = 0
    timepre1 = time.time()

    for (region_x, region_y), confocal in zip(regions_selected, confocal_stacks[0]):
        time_region = time.time()

        confocal = (confocal/confocal.max() - 0.5) / 0.5

        # fix relative offset
        region_x -= width/2
        region_y -= height/2

        # simplify confidence map to regions of predefined size ((px,px))
        sup_px_map = skimage.measure.block_reduce(std_map_allROI[ROI_idx], (px,px), numpy.mean)  # nanmean or nansum
        sup_px_map = Image.fromarray(sup_px_map).resize((confocal.shape[1], confocal.shape[0]),resample=Image.NEAREST)
        sup_px_map = numpy.array(sup_px_map)

        # use ROI_map to avoid taking STED in the centered region of interest
        sup_px_map = (sup_px_map) * (1-ROI_map)# * (1-(already_acquired[ROI_idx]+1)/2)
        sup_px_map_allROI[ROI_idx] = sup_px_map
        # binarize (0 everywhere except in region where we want to acquire STED)
        sup_px_map = (sup_px_map == sup_px_map.max()).astype('uint8')
        h, w = sup_px_map.shape
        mgrid = numpy.mgrid[0:h:px, 0:w:px].reshape(2, -1).T
        rect_regions_offset, rect_regions_sub_offset, regions = [], [], []
        for pair in mgrid:
            x, y = pair
            if sup_px_map[x+int(px/2), y+int(px/2)]:
                rect_regions_offset.append([region_x + y * 20*1e-9 + px * 20*1e-9 / 2, region_y + x * 20*1e-9 + px * 20*1e-9 / 2])
                rect_regions_sub_offset.append([x * 20*1e-9 + px * 20*1e-9 / 2, y * 20*1e-9 + px * 20*1e-9 / 2])
                regions.append((px * 20*1e-9 , (px+1) * 20*1e-9))
                break # once one region is acquired, we don't need to go over the others since we just take one

        # Acquire STED crops
        timepre=time.time()
        print('Imaging %s super pixels out of %s' % (len(regions), len(mgrid)))
        t, _, STED_stacks = run_STED(config_STED, rect_regions_offset, regions, ROI_idx, output, stedimsavepath, rep=r) # regions_for_sted

        timepost=time.time()
        elapsed=timepost-timepre
        print('Acquiring one STED crop took %s seconds.' % elapsed)

        # normalize using absolute 0-255 scale and add 0.5
        STED_stacks_norm = (numpy.array(STED_stacks[0]) / 255.0 - 0.5) / 0.5 + 0.5
        STED_stacks_norm = STED_stacks_norm[:,:px,:px]

        # Place the STED crops inside the sup_px_map
        already_acquired = numpy.zeros((len(regions_selected), h, w)) - 1
        for (x, y), patch_norm, patch in zip(rect_regions_sub_offset, STED_stacks_norm, STED_stacks[0]):
            x, y = int((x) / (20*1e-9)), int((y) / (20*1e-9))
            sup_px_map_STED[ROI_idx, x-int(px/2):x+int(px/2), y-int(px/2):y+int(px/2)] = patch_norm
            already_acquired[ROI_idx,x-int(px/2):x+int(px/2), y-int(px/2):y+int(px/2)] = 1

        # Predict synthetic STED
        timepre = time.time()
        sup_px_map_for_input = torch.from_numpy(sup_px_map_STED[ROI_idx]) + 0.5
        sup_px_map_for_input *= (torch.from_numpy(already_acquired[ROI_idx]+1)/2)
        #sup_px_map_for_input[already_acquired[ROI_idx] == -1] = -1

        data = torch.cat((torch.from_numpy(confocal).unsqueeze(0), sup_px_map_for_input.unsqueeze(0), torch.from_numpy(already_acquired[ROI_idx]).unsqueeze(0)), 0)

        tifffile.imsave(os.path.join(output,'input{}_ROI{}.tif'.format(r,ROI_idx)), numpy.array(data).astype('float32'))
        model = VirtualNet(data, full_STED_stacks[0][ROI_idx]) # Here sup_px_map is between -1 and 1
        std_map, synthetic, next_STED, seg_real, seg_fake = model.predict()
        std_map_allROI[ROI_idx] = std_map * (1-ROI_map)# * (1-(already_acquired[ROI_idx]+1)/2)
        synthetic_allROI[ROI_idx] = synthetic

        print('Computing time on server: {}'.format(time.time() - timepre))
        print('Total time for one region: {}'.format(time.time() - time_region))

        # Check if we should acquired full STED
        if next_STED:
            _, new_STED_stacks, _, _ = run(config_STED, 0, [[region_x + width / 2, region_y + width / 2]], output, stedimsavepath, rep=r)
            print(numpy.shape(full_STED_stacks), numpy.shape(new_STED_stacks))
            full_STED_stacks[0][ROI_idx] = new_STED_stacks[0][0]
            tifffile.imsave(os.path.join(output,'fullSTED{}_ROI{}.tif'.format(r,ROI_idx)), numpy.array(full_STED_stacks).astype('uint16'))
            tifffile.imsave(os.path.join(output,'fullSTED_seg{}_ROI{}.tif'.format(r,ROI_idx)), numpy.array((seg_real+1)/2*255).astype('uint8'))

        tifffile.imsave(os.path.join(output,'synthetic{}_seg_allROI{}.tif'.format(r,ROI_idx)), ((numpy.transpose(seg_fake,(2,0,1))+1)/2*255).astype('uint8'))

        ROI_idx += 1

    tifffile.imsave(os.path.join(output,'synthetic_allROI{}.tif'.format(r)), ((synthetic_allROI+1)/2*255).astype('uint8'))

    if tlim-(time.time()-timepre1) > 0:
        print('Sleeping for {}'.format(time.time()-timepre1))
        time.sleep(tlim-(time.time()-timepre1))

# Final STED
_, STED_stacks, _, _ =  run(config_STED, 0, regions_selected, output, stedimsavepath, rep=repetitions+1)
tifffile.imsave(os.path.join(output,'finalSTED.tif'), numpy.array(STED_stacks).astype('uint16'))

# Final confocal
_, confocal_stacks, _, _ = run(config_confocal, 0, regions_selected, output, confimsavepath, rep=repetitions+1)
tifffile.imsave(os.path.join(output,'finalconfocal.tif'), numpy.array(confocal_stacks).astype('uint16'))
