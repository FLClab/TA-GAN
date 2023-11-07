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
import microscope_newspecpy as microscope
import user
import utils
from datetime import date

from options.test_options import TestOptions
from models import create_model
import util
import torch
from PIL import Image
import skimage.measure
import skimage.filters
from torchvision import transforms
import tifffile

import requests
from skimage.filters import threshold_triangle, threshold_otsu, gaussian

name = 'Auto_Select_MultiControl'
id=input('Enter sample region identifier: ')

px = 100# pixel size (min. 50)
ROI = 300 # center region of interest where no STED will be acquired (excpet full STED)
width, height = 500.0*20*1e-9, 501.0*20*1e-9

config_STED = microscope.get_config("Setting STED configuration.")
config_confocal = microscope.get_config("Setting Confocal configuration.")

regions_selected = user.get_regions()

def run(config_window, t, regions, output, imsavepath = None,rep=1):
    #  Show overview image to user and ask for rectangle selections
    overview = '640'
    width, height = 500.0*20*1e-9, 501.0*20*1e-9

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
        print(width, height)
        microscope.set_imagesize(config_window, width, height)

        # Set microscope configurations to set RESCue parameters
        print('microscope parameters set')

        # This will output a .csv file for every region that were imaged
        # parameters x,y are the imaging region offset center
        # width, height are the imaging size
        with open(os.path.join(imsavepath, "imaging_parameters_ROI{}.csv".format(t)), "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([x, y, width, height])
        # Acquire image and save file to .msr and images to .tiff

        timepre = time.time()
        stacks, _ = microscope.acquire_saveasmsr(config_window, os.path.join(imsavepath, "_ROI{}_{}.msr".format(t,rep)))
        print('Acquiring the confocal took {} seconds'.format(time.time()-timepre))
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
        t += 1
        #!!!print('regions saved to csv')

        # Set microscope configurations to image selected region
        microscope.set_offsets(config_window, x, y)
        print(width, height)
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
        stacks, _ = microscope.acquire_saveasmsr(config_window, os.path.join(imsavepath, "_ROI{}_{}.msr".format(t,rep)))
        if t==1:
            all_stacks = stacks
        else:
            all_stacks = numpy.concatenate([all_stacks, stacks], 1)
        #stacks_list.append(stacks[1][0]) # !!! Pourquoi [0] ? C'est quoi l'autre channel?

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

    def __init__(self, img, decision_map, S, address='172.16.11.2', port=5000):
        self.img = img
        self.decision_map = decision_map
        self.S = S
        self.address = address
        self.port = port
        self.url = 'http://{}:{}'.format(self.address, self.port)
        print('Running on server '+self.url)

    def predict(self):
        """Generate a fake STED from confocal input
        """
        tosend = json.dumps({'INPUT':self.img.tolist(), 'DECISION_MAP':self.decision_map.tolist(), 'S':self.S.tolist()})
        r = requests.post(self.url, data=tosend)
        data = json.loads(r.text)
        return numpy.array(data['std_map']), numpy.array(data['synthetic_STED'][0][0])

def update(val):
    thresh = sap.val
    sup_px_map_thresh = (sup_px_map > thresh).astype(int)
    im.set_data(sup_px_map_thresh)
    im.axes.figure.canvas.draw()

# Acquire Confocal
timepre1=time.time()
t, confocal_stacks, _, _ = run(config_confocal, 0, regions_selected, output, confimsavepath, rep=0)
timepost1=time.time()
elapsed1=timepost1-timepre1
print(numpy.shape(confocal_stacks))
tifffile.imsave(os.path.join(output,'initialconfocal.tif'), numpy.array(confocal_stacks).astype('uint16'))
time.sleep(1)

# acquire initial STED
regions = [[(width, height)]*len(regions_selected)]
timepre1 = time.time()
_, STED_stacks, _, _ =  run(config_STED, 0, regions_selected, output, confimsavepath, rep=0)

print('Acquiring the complete STED took {} seconds'.format(time.time() - timepre1))
STED_stacks = numpy.array(STED_stacks).astype('uint16')
print(STED_stacks.shape)
tifffile.imsave(os.path.join(output,'initialSTED.tif'), STED_stacks)
time.sleep(1)

repetitions = 30
tlim = 60

for r in range(1,repetitions):
    # Acquire STED
    timepre=time.time()
    t, STED_stacks, _, _ = run(config_STED, 0, regions_selected, output, stedimsavepath, rep=r)
    tifffile.imsave(os.path.join(output,'STED{}.tif'.format(r)), numpy.array(STED_stacks).astype('uint16'))

    region = 0
    for (region_x, region_y), confocal in zip(regions_selected, STED_stacks):
        print('Sleeping for {} seconds (repetition {}/{}, region {}/{})'.format(tlim, r, repetitions, region, len(regions_selected)))
        time.sleep(21)
        region += 1

    if tlim-(time.time()-timepre1) > 0:
        print('Sleeping for {}'.format(time.time()-timepre1))
        time.sleep(tlim-(time.time()-timepre1))

# Final confocal
_, confocal_stacks, _, _ = run(config_confocal, 0, regions_selected, output, confimsavepath, rep=repetitions)
tifffile.imsave(os.path.join(output,'finalconfocal.tif'), numpy.array(confocal_stacks).astype('uint16'))

# Final STED
_, STED_stacks, _, _ =  run(config_STED, 0, regions_selected, output, stedimsavepath, rep=repetitions)
tifffile.imsave(os.path.join(output,'finalSTED.tif'), numpy.array(STED_stacks).astype('uint16'))
