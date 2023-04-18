import tifffile
import numpy
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

plt.figure(figsize=(3,4))
TAGAN_color = '#640064'
STED_color = '#ffa100'

mean_list = numpy.load('20211201_cs5_ROI1_fibers.npy')
sted_list = numpy.load('20211201_cs5_ROI1_fibers_STED.npy')

plt.scatter(numpy.arange(1,16), mean_list, c=TAGAN_color, edgecolor='k', linewidth=2, s=100, label='Synthetic')

idx_sted = [1,2,3,14,15]
for j, i in enumerate(idx_sted):
    ## Add dashed lines to connect real and synthetic STED when STED was acquired
    plt.plot([i,i], [sted_list[j], mean_list[i-1]], '--', color='#aaaaaa')

    ## Replace value of TA-GAN by STED value for plotting the progression line
    mean_list[i-1] = sted_list[j]

# Plot STED points
plt.scatter(idx_sted, sted_list, c=STED_color, edgecolor='k', linewidth=2, s=100, label='STED')

# Plot progression line
plt.plot(numpy.arange(1,16), mean_list, '--k')

plt.xlabel('Time (min.)')
plt.ylabel('Proportion of fibers in dendrite')
plt.axis([-1,16,0,0.5])
plt.savefig('fibers_prop_20211201_cs5_roi1.pdf')
plt.show()

