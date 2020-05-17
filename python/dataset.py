#!/usr/bin/env python

"""
PyTorch datasets and data augmenters
"""

###########
# Imports #
###########

import cv2
import numpy as np
import os
import random
import torch

from PIL import Image, ImageFilter
from torch.utils import data
from torchvision import transforms


#############
# Functions #
#############

def to_pil(tensor):
	'''Converts a tensor to a PIL image.'''
	return transforms.functional.to_pil_image(tensor)


def to_tensor(pic):
	'''Converts a PIL image to a tensor.'''
	return transforms.functional.to_tensor(pic)


def to_mask(shape, polygons):
	'''Builds a mask based on polygon annotations.'''
	contours = [np.array(p, dtype=int) for p in polygons]

	mask = np.zeros(shape, dtype=np.uint8)
	cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

	return Image.fromarray(mask)


def to_polygons(mask, threshold=128):
	'''Converts a mask into polygon annotations.'''
	mask = np.array(mask)

	_, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	polygons = [c[:, 0, :].tolist() for c in contours]

	return polygons


def to_contours(mask):
	'''Converts a mask into OpenCV contours.'''
	mask = np.array(mask)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	return contours


def clusterize(polygons, size):
	'''Clusterize polygons.'''
	clusters = {}

	for polygon in polygons:
		temp = np.array(polygon).astype(int)

		xmin = np.amin(temp[:, 0]) // size
		xmax = np.amax(temp[:, 0]) // size
		ymin = np.amin(temp[:, 1]) // size
		ymax = np.amax(temp[:, 1]) // size

		for x in range(xmin, xmax + 1):
			for y in range(ymin, ymax + 1):
				key = x * size, y * size

				if not key in clusters:
					clusters[key] = []

				clusters[key].append(polygon)

	return clusters


###########
# Classes #
###########

class VIADataset(data.IterableDataset):
	'''Iterable VIA dataset.'''

	def __init__(self, via, path='./', size=256, shuffle=False, shift=0, full=False, alt=0):
		self.via = {}
		self.masks = {}
		self.clusters = {}

		self.size = size

		for key, polygons in via.items():
			imagename = os.path.join(path, key)

			if os.path.exists(imagename):
				image = Image.open(imagename)

				self.via[imagename] = polygons
				self.masks[imagename] = to_mask((image.height, image.width), polygons)

				if self.size is not None:
					self.clusters[imagename] = clusterize(polygons, self.size)

		self.shuffle = shuffle # random order
		self.shift = shift # random shift
		self.full = full # all sub-images
		self.alt = alt # alternate

	def __len__(self):
		if self.size is None:
			return len(self.via)
		elif self.full:
			s = 0
			for imagename in self.via:
				image = Image.open(imagename)
				s += (image.width // self.size) * (image.height // self.size)
			return s
		else:
			return sum(map(len, self.clusters.values())) * (1 + self.alt)

	def __iter__(self):
		images = random.sample(
			self.via.keys(),
			len(self.via)
		) if self.shuffle else self.via.keys()

		for imagename in images:
			image = Image.open(imagename).convert('RGB')
			mask = self.masks[imagename]

			if self.size is None:
				yield image, mask
			elif self.full:
				for left in np.arange(0, image.width, self.size):
					for upper in np.arange(0, image.height, self.size):
						box = (left, upper, left + self.size, upper + self.size)
						yield image.crop(box), mask.crop(box)
			else:
				clusters = list(self.clusters[imagename].keys())

				if self.shuffle:
					random.shuffle(clusters)

				for left, upper in clusters:
					# Shift
					if self.shift > 0:
						left += random.randint(-self.shift, self.shift)
						upper += random.randint(-self.shift, self.shift)

					# Out of bounds
					left = min(left, image.width - self.size)
					upper = min(upper, image.height - self.size)

					box = (left, upper, left + self.size, upper + self.size)

					yield image.crop(box), mask.crop(box)

					# Alternate with random images
					for _ in range(self.alt):
						left = random.randrange(image.width - self.size)
						upper = random.randrange(image.height - self.size)

						box = (left, upper, left + self.size, upper + self.size)

						yield image.crop(box), mask.crop(box)


class RandomChoice(data.IterableDataset):
	'''Apply a randomly picked transformation to each pair (input, target).'''

	def __init__(self, dataset, transforms, input_only=False):
		super().__init__()

		self.dataset = dataset
		self.transforms = transforms
		self.input_only = input_only

	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		for input, target in self.dataset:
			f = random.choice(self.transforms)
			yield f(input), target if self.input_only else f(target)


class ColorJitter(RandomChoice):
	'''Color jitter.'''

	def __init__(self, dataset, brightness=0.25, contrast=0.33, saturation=0.33, hue=0):
		super().__init__(
			dataset=dataset,
			transforms=[transforms.ColorJitter(brightness, contrast, saturation, hue)],
			input_only=True
		)


class RandomFilter(RandomChoice):
	'''Random image filter.'''

	def __init__(self, dataset):
		super().__init__(
			dataset=dataset,
			transforms=[
				lambda x: x,
				lambda x: x.filter(ImageFilter.BLUR),
				lambda x: x.filter(ImageFilter.DETAIL),
				lambda x: x.filter(ImageFilter.EDGE_ENHANCE),
				lambda x: x.filter(ImageFilter.SMOOTH),
				lambda x: x.filter(ImageFilter.SHARPEN)
			],
			input_only=True
		)


class RandomTranspose(RandomChoice):
	'''Random image transpose.'''

	def __init__(self, dataset):
		super().__init__(
			dataset=dataset,
			transforms=[
				lambda x: x,
				lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
				lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
				lambda x: x.transpose(Image.ROTATE_90),
				lambda x: x.transpose(Image.ROTATE_180),
				lambda x: x.transpose(Image.ROTATE_270),
				lambda x: x.transpose(Image.TRANSPOSE)
			],
			input_only=False
		)


class Scale(RandomChoice):
	'''Scale image.'''
	def __init__(self, dataset, scale):
		super().__init__(
			dataset=dataset,
			transforms=[lambda x: x.resize(
				(int(x.width * scale), int(x.height * scale))
			)],
			input_only=False
		)


class ToTensor(RandomChoice):
	'''To Tensor.'''

	def __init__(self, dataset):
		super().__init__(
			dataset=dataset,
			transforms=[to_tensor],
			input_only=False
		)


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import json
	import via as VIA

	# Arguments
	parser = argparse.ArgumentParser(description='Format California annotations to the VIA format')
	parser.add_argument('-e', '--ext', default='.tif', help='extension of the images')
	parser.add_argument('-o', '--output', default='../products/json/california.json', help='output VIA file')
	parser.add_argument('-p', '--path', default='../resources/california/', help='path to California resources')
	args = parser.parse_args()

	# Polygons
	with open(os.path.join(args.path, 'SolarArrayPolygons.json'), 'r') as f:
		panels = json.load(f)['polygons']

	# VGG Image Annotations
	via = {}

	for panel in panels:
		filename = panel['image_name'] + args.ext
		polygon = panel['polygon_vertices_pixels']

		## Skip dots and lines
		if not len(polygon) > 3:
			continue

		## Add polygon
		if filename not in via:
			via[filename] = []

		via[filename].append(polygon)

	# Save
	VIA.dump(via, args.output, path=args.path)
