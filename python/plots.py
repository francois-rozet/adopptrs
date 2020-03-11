"""
"""

###########
# Imports #
###########

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import LargeDataset


#############
# Functions #
#############

def flatten(array, depth=1):
	'''Flattens @array into a list up to depth @depth.'''
	if depth == 0:
		return array
	else:
		l = list()

		for i in array:
			l.extend(flatten(i, depth - 1))

		return l

def plot_images(images, ncols=2, zoom=4):
	'''Plots a list of images as a grid with @ncols columns.'''
	nrows = len(images) // ncols

	# Initialize grid
	fig, ax = plt.subplots(
		nrows, ncols,
		sharex=True, sharey=True,
		figsize=(ncols * zoom, nrows * zoom)
	)

	# Plot images
	for i in range(len(images)):
		pil = LargeDataset.revert(images[i])

		if pil.mode == 'L':
			ax[i // ncols, i % ncols].imshow(pil, cmap=cm.gray)
		else:
			ax[i // ncols, i % ncols].imshow(pil)

def plot_alongside(*argv):
	'''Plots image lists alongside.'''
	plot_images(flatten([list(t) for t in zip(*argv)]), len(argv))
