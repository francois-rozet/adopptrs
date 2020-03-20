"""
"""

###########
# Imports #
###########

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import to_pil


#############
# Functions #
#############

def flatten(array, n=1):
	'''Flattens an array into a list.'''
	if n == 0:
		return array
	else:
		l = list()

		for i in array:
			l.extend(flatten(i, n - 1))

		return l


def plot_images(images, ncols=2, zoom=4):
	'''Plots a list of images as a grid.'''
	nrows = len(images) // ncols

	# Initialize grid
	fig, ax = plt.subplots(
		nrows, ncols,
		sharex=True, sharey=True,
		figsize=(ncols * zoom, nrows * zoom)
	)

	# Plot images
	for i in range(len(images)):
		ax[i // ncols, i % ncols].imshow(to_pil(images[i]))


def plot_alongside(*argv):
	'''Plots image lists alongside.'''
	plot_images(
		flatten(map(lambda x: list(x), zip(*argv))),
		len(argv)
	)
