import tifffile
import os
import numpy
import matplotlib.pyplot as plt
import scipy
from skimage.transform import rescale
from skimage.morphology import binary_dilation, disk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tqdm

import matplotlib

font = {'size'   : 22}

matplotlib.rc('font', **font)


def dice(im1, im2):

    # Compute Dice coefficient
    intersection = numpy.logical_and(im1, im2)
    union = im1 + im2

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

folders = ['SIM', 'BF','pix2pix', 'tagan_LRlabels', 'tagan_precise']
color_list = ['k', 'r', 'b', 'g', 'm']

t = 185
tBF = 12437 # Threshold obtained from precision recall curve
pred_t = 20 
vis = False
registration = True

for i, folder in enumerate(folders):
	
	label_list = numpy.load('{}_labels.npy'.format(folder))
	pred_list = numpy.load('{}_preds.npy'.format(folder))

	print(len(label_list))

	cm = confusion_matrix(label_list, pred_list)
	dc = (2*cm[1,1])/(2*cm[1,1]+cm[0,1]+cm[1,0])
	acc = (cm[0,0]+cm[1,1])/cm.sum()
	print(cm, dc, acc)
	print('Class 0 / class 1: {} / {}'.format(cm[0,0]+cm[0,1], cm[1,0]+cm[1,1]))

		#plt.scatter(label_list, pred_list)
		#plt.show()

	disp = ConfusionMatrixDisplay.from_predictions(label_list, pred_list, cmap='Blues', normalize='true', colorbar=False, display_labels=['Non-Dividing','Dividing'])
	disp.ax_.get_images()[0].set_clim(0, 1)
	plt.tight_layout()
	plt.savefig(f"{folder}.pdf")
	plt.savefig(f"{folder}.png")
