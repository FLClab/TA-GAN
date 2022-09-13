import numpy
import tifffile
import os

input_folder_top = ''		# Enter folder name
input_folder_bottom = ''	# Enter folder name
output_folder = ''			# Enter folder name

for file in os.listdir(input_folder_top):
	top = tifffile.imread(os.path.join(input_folder_top,file))
	bottom = tifffile.imread(os.path.join(input_folder_bottom,file))
	combined = numpy.concatenate([top, bottom], axis=1)
	tifffile.imsave(os.path.join(output_folder,file), combined)
