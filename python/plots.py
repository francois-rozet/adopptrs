#!/usr/bin/env python

"""
Image plotting helpers
"""

###########
# Imports #
###########

import matplotlib.pyplot as plt
import matplotlib.cm as cm


#############
# Functions #
#############

def flatten(array, n=1):
	'''Flattens an array into a list.'''
	if n == 0:
		return array
	else:
		l = []

		for sub in array:
			l.extend(flatten(sub, n - 1))

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
		ax[i // ncols, i % ncols].imshow(images[i])


def plot_alongside(*argv, zoom=4):
	'''Plots image lists alongside.'''
	plot_images(
		flatten(map(list, zip(*argv))),
		len(argv),
		zoom
	)
