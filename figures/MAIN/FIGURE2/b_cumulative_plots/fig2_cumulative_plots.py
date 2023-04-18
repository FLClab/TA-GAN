import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib
import os

font = {'family' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

pair = 'PSDBassoon'
N = 9

xlabels = ['Eccentricity','Area', 'Distance to neighbor same ch', 'Distance to neighbor other ch', 'Perimeter', 'Coupling Probability']
range_maxs = [1.1, 700*0.015**2, 105, 105, 255, 1]
range_steps = [0.1, 10*0.015**2, 1, 1, 1, 0.1]
columns = [5,2,3,4,12,14]
titles = ['eccentricity', 'area', 'distance_to_same', 'distance_to_other', 'perimeter', 'coupling probability']	


for xlabel, range_max, range_step, column, title in zip(xlabels, range_maxs, range_steps, columns, titles):

	fig, axs = plt.subplots(2,1,figsize=(5,6))
	print('Making figure for '+xlabel+'...')

	for ch in [0,1]:

		for folder, label, color in zip(['STED','CONF','P2P','LOC','SEG'],['STED','Confocal','pix2pix','TAGAN Loc.','TAGAN Seg.'],['#ff9900','#ff0000','#0000ff','#9900cc','#cc00cc']):
			i = 0
			all_spots = numpy.zeros((N,int(range_max/range_step)+1))
			all_values = numpy.array([])
			for file in os.listdir(folder):
				if '.xlsx' in file[-5:]:
					file_name = os.path.join(folder,file)
					df = pandas.read_excel(file_name, sheet_name="Spots ch{}".format(ch))

					df_np = df.to_numpy()

					if xlabel == 'Area': # convert px^2 to um^2
						df_np[:,column] *= 0.015**2

					all_values = numpy.append(all_values, df_np[:,column])

					values, base = numpy.histogram(df_np[:,column], numpy.linspace(-range_step,range_max,int(range_max/range_step+2)), density=True)

					all_spots[i,:] = numpy.cumsum(values) / numpy.max(numpy.cumsum(values))

					i += 1

			axs[ch].plot(base[:-1], numpy.mean(all_spots,0), c=color, label=label)
			axs[ch].fill_between(base[:-1], numpy.mean(all_spots,0)-numpy.std(all_spots,0), numpy.mean(all_spots,0)+numpy.std(all_spots,0), color=color, alpha=0.2)
			se = numpy.std(all_spots, ddof=1, axis=0)

		axs[ch].set_xlabel(xlabel)
		axs[ch].set_ylabel('Cumulative\nfrequency')
		axs[ch].set_yticks([0,0.5,1])
		#axs[ch].set_xticks([0.05,0.55,0.06])
		#axs[ch].set_xlim(0.05, 0.06) ## Zommed-in in-set
		#axs[ch].set_ylim(0.65,0.875)

	#plt.legend()
	plt.tight_layout()
	plt.savefig('computed_stats/{}_{}.pdf'.format(pair, title))
	plt.savefig('computed_stats/{}_{}.png'.format(pair, title))
	plt.show()

	plt.close()