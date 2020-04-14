"""
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
	contours = [np.array(p, dtype=int) for p in polygons]

	mask = np.zeros(shape, dtype=np.uint8)
	cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

	return Image.fromarray(mask)


def to_polygons(mask):
	mask = np.array(mask)

	_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	polygons = [c[:, 0, :].tolist() for c in contours]

	return polygons


###########
# Classes #
###########

class VIADataset(data.IterableDataset):
	'''Iterable VIA dataset.'''

	def __init__(self, via, path='./', size=256, shuffle=False):
		self.via = {
			os.path.join(path, key): value for key, value in via.items()
			if os.path.exists(os.path.join(path, key))
		}

		self.images = list(self.via.keys())
		self.masks = {}

		self.size = size

		self.shuffle = shuffle

	def __len__(self):
		if self.size is None:
			return len(self.via)
		else:
			return sum(map(len, self.via.values()))

	def __iter__(self):
		if self.shuffle:
			random.shuffle(self.images)

		for imagename in self.images:
			image = Image.open(imagename).convert('RGB')

			if imagename not in self.masks:
				self.masks[imagename] = to_mask((image.width, image.height), self.via[imagename])

			mask = self.masks[imagename]

			if self.size is None:
				yield image, mask
			else:
				if self.shuffle:
					random.shuffle(self.via[imagename])

				for polygon in self.via[imagename]:
					left, upper = random.choice(polygon)

					# Transpose
					left -= random.randrange(self.size)
					upper -= random.randrange(self.size)

					# Box
					left = max(left, 0)
					left = min(left, image.width - self.size)

					upper = max(upper, 0)
					upper = min(upper, image.height - self.size)

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
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--ext', default='.tif', help='extension of the images')
	parser.add_argument('-o', '--output', default='../products/json/california.json', help='output VIA file')
	parser.add_argument('-p', '--path', default='../resources/california/', help='path to california resources')
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
