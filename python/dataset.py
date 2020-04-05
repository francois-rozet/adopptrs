"""
"""

###########
# Imports #
###########

import json
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


###########
# Classes #
###########

class LargeDataset(data.IterableDataset):
	'''Iterable dataset for large images.'''

	def __init__(self, data, in_ram=False, shuffle=False):
		super().__init__()

		self.data = data
		self._len = 0

		self.in_ram = in_ram
		if self.in_ram:
			self.data = [
				(Image.open(x).convert('RGB'), Image.open(y), z)
				for x, y, z in self.data
			]

		self.shuffle = shuffle

	def __len__(self):
		if self._len == 0:
			for _, _, boxes in self.data:
				self._len += len(boxes)

		return self._len

	def __iter__(self):
		if self.shuffle:
			random.shuffle(self.data)

		for image, mask, boxes in self.data:
			if not self.in_ram:
				image = Image.open(image).convert('RGB')
				mask = Image.open(mask)

			if self.shuffle:
				random.shuffle(boxes)

			for box in boxes:
				yield image.crop(box), mask.crop(box)


	@staticmethod
	def load(filename, path=''):
		'''Load data file.'''
		with open(filename, 'r') as f:
			data = [
				(os.path.join(path, x), os.path.join(path, y), box)
				for x, y, box in json.load(f)
			]

		return data


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
				lambda x: x.filter(ImageFilter.EDGE_ENHANCE_MORE),
				lambda x: x.filter(ImageFilter.SMOOTH),
				lambda x: x.filter(ImageFilter.SMOOTH_MORE),
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


class ToTensor(RandomChoice):
	'''To Tensor.'''

	def __init__(self, dataset):
		super().__init__(
			dataset=dataset,
			transforms=[transforms.ToTensor()],
			input_only=False
		)


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import cv2
	import numpy as np

	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--destination', default='../products/json/', help='destination of the file listing image-mask pairs')
	parser.add_argument('-e', '--ext', default='.tif', help='extension of the images')
	parser.add_argument('-n', '--name', default='data.json', help='name of the file listing image-mask pairs')
	parser.add_argument('-m', '--masks', default=False, action='store_true', help='recompute masks')
	parser.add_argument('-o', '--output', default=None, help='standard output file')
	parser.add_argument('-p', '--path', default='../resources/', help='path to resources')
	parser.add_argument('-s', '--size', type=int, default=256, help='subimages size')
	args = parser.parse_args()

	# Log file
	if args.output is not None:
		sys.stdout = open(args.output, 'a')

	# Polygons
	with open(os.path.join(args.path, 'polygons/SolarArrayPolygons.json'), 'r') as f:
		polygons = json.load(f)['polygons']

	# Table {imagename: [polygon, ...], ...}
	table = {}

	for p in polygons:
		imagename = p['city'] + '/' + p['image_name']
		contour = np.array(p['polygon_vertices_pixels'], dtype=int)

		## Skip dots and lines
		if contour.shape[0] <= 2:
			continue

		## Add to table
		if imagename in table:
			table[imagename].append(contour)
		else:
			table[imagename] = [contour]

	# Masks and subimages
	data = []

	for imagename, contours in table.items():
		if not os.path.exists(os.path.join(args.path, imagename + args.ext)):
			continue

		print('-' * 10)
		print('Loading {}'.format(imagename + args.ext))

		maskname = imagename + '_mask'

		if os.path.exists(os.path.join(args.path, maskname + args.ext)) and not args.masks:
			print('Loading {}'.format(maskname + args.ext))

			mask = Image.open(os.path.join(args.path, maskname + args.ext))
			mask = np.array(mask).astype(np.uint8)
		else:
			print('Creating {}'.format(maskname + args.ext))

			## Get image shape
			image = Image.open(os.path.join(args.path, imagename + args.ext))
			mask = np.zeros((image.height, image.width), np.uint8)

			## Draw polygons interior
			cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

			## Save mask
			cv2.imwrite(os.path.join(args.path, maskname + args.ext), mask)

		## Subimage locations
		boxes = []

		for x in range(0, mask.shape[0] - args.size, args.size // 2):
			for y in range(0, mask.shape[1] - args.size, args.size // 2):
				if np.any(mask[x:x+args.size, y:y+args.size]):
					boxes.append((y, x, y+args.size, x+args.size))

		data.append((
			imagename + args.ext,
			maskname + args.ext,
			boxes
		))

		print('Found {} boxes'.format(len(boxes)))

	# mkdir -p destination
	os.makedirs(args.destination, exist_ok=True)

	with open(os.path.join(args.destination, args.name), 'w') as f:
		json.dump(data, f)
