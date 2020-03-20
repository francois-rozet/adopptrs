"""
"""

###########
# Imports #
###########

import random
import torch

from PIL import Image
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

	def __init__(self, data, transform=None, color=None):
		super().__init__()

		self.data = data
		self._len = 0

		if transform is None:
			self.transform = lambda x: x
		elif transform == 'tensor':
			self.transform = transforms.ToTensor()
		else:
			self.transform = transform

		if color is None:
			self.color = lambda x: x
		elif color == 'jitter':
			self.color = transforms.ColorJitter(0.25, 0.33, 0.33)
		else:
			self.color = color

	def __len__(self):
		if self._len == 0:
			for _, _, boxes in self.data:
				self._len += len(boxes)

		return self._len

	def __iter__(self):
		for imagename, maskname, boxes in self.data:
			image = Image.open(imagename)
			mask = Image.open(maskname)

			image = self.color(image.convert('RGB'))

			for box in boxes:
				yield (
					self.transform(image.crop(box)),
					self.transform(mask.crop(box))
				)

	def shuffle(self):
		'''Shuffles dataset.'''
		random.shuffle(self.data)
		for _, _, boxes in self.data:
			random.shuffle(boxes)


class RandomTurn(data.IterableDataset):
	'''Iterable random dataset turner.'''

	flip = [
		lambda x: x,
		lambda x: x.flip(1),
		lambda x: x.flip(2)
	]

	rotate = [
		lambda x: x,
		lambda x: x.rot90(1, [1, 2]),
		lambda x: x.rot90(2, [1, 2]),
		lambda x: x.rot90(3, [1, 2])
	]

	def __init__(self, dataset):
		super().__init__()

		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		for input, target in self.dataset:
			f = random.choice(self.flip)
			g = random.choice(self.rotate)

			yield g(f(input)), g(f(target))

	def __getattr__(self, name):
		return getattr(self.dataset, name)


########
# Main #
########

if __name__ == '__main__':
	'''Creation of masks and computation of subimage locations.'''

	# Imports
	import cv2
	import json
	import numpy as np
	import os

	# Parameters
	origin = 'resources/'
	destination = 'products/json/'
	ext = '.tif'

	size = 256
	step = 128

	# Polygons
	with open(origin + 'polygons/SolarArrayPolygons.json', 'r') as f:
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
		if not os.path.exists(origin + imagename + ext):
			continue

		## Get image shape
		image = Image.open(origin + imagename + ext)
		mask = np.zeros((image.height, image.width), np.uint8)

		## Draw polygons interior
		cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

		## Save mask
		maskname = imagename + '_mask'
		cv2.imwrite(origin + maskname + ext, mask)

		## Subimage locations
		boxes = []

		for x in range(0, mask.shape[0] - size, step):
			for y in range(0, mask.shape[1] - size, step):
				if np.any(mask[x:x+size, y:y+size]):
					boxes.append((y, x, y+size, x+size))

		data.append((
			imagename + ext,
			maskname + ext,
			boxes
		))

	# mkdir -p destination
	os.makedirs(destination, exist_ok=True)

	with open(destination + 'data.json', 'w') as f:
		json.dump(data, f)
