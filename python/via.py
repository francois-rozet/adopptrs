#!/usr/bin/env python

"""
VGG Image Annotator package

References
----------
The VIA annotation software for images, audio and video
(Dutta et al., 2019)
https://arxiv.org/abs/1904.10699
"""

###########
# Imports #
###########

import json
import os


#############
# Functions #
#############

def load(filename):
	with open(filename, 'r') as f:
		return deformat(json.load(f))


def dump(via, filename, path='./', indent=None):
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	with open(filename, 'w') as f:
		json.dump(format(via, path), f, indent=indent)


def deformat(via):
	'''Deformats VIA dictionnary'''
	deformated = {}

	for file in via.values():
		polygons = []

		for region in file['regions']:
			polygons.append(list(zip(
				region['shape_attributes']['all_points_x'],
				region['shape_attributes']['all_points_y']
			)))

		deformated[file['filename']] = polygons

	return deformated


def format(via, path='./'):
	'''Format VIA dictionnary'''
	formated = {}

	for basename, polygons in via.items():
		filename = os.path.join(path, basename)

		if os.path.exists(filename):
			size = os.stat(filename).st_size
		else:
			size = 0

		formated[basename + str(size)] = {
			'filename': basename,
			'size': size,
			'file_attributes': {},
			'regions': list(map(
				lambda polygon: {
					'region_attributes':{},
					'shape_attributes': {
						'all_points_x': list(map(lambda x: x[0], polygon)),
						'all_points_y': list(map(lambda x: x[1], polygon)),
						'name': 'polygon'
					}
				},
				polygons
			))
		}

	return formated
