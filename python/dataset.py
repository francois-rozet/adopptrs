"""
"""

###########
# Imports #
###########

import numpy as np
import os.path

from PIL import Image
from torch.utils import data
from torchvision import transforms


#############
# Functions #
#############

def shift(image, x, y):
	'''Removes the @x first rows and @y first columns of @image.'''
	return image[x:, y:, :]

def rotate(image, angle):
	'''Rotates @image of @angle degrees around its center.'''
	height, width = image.shape[:2]
	center = height / 2, width / 2
	scale = 1

	# Rotation matrix
	matrix = cv2.getRotationMatrix2D(center, angle, scale)

	# New dimensions
	alpha = np.radians(angle)
	h = height * np.cos(alpha) + width * np.sin(alpha)
	w = height * np.sin(alpha) + width * np.cos(alpha)

	return cv2.warpAffine(image, matrix, (int(h), int(w)))


###########
# Classes #
###########

class LargeDataset(data.IterableDataset):
	'''Iterable dataset for large images.'''

	def __init__(self, images, random_state=None):
		self.images = images

		if random_state is not None:
			np.random.seed(random_state)
		np.random.shuffle(self.images)

		self.start, self.end = 0, len(self.images)

	def __len__(self):
		return len(self.images)

	def __iter__(self):
		worker = data.get_worker_info()

		if worker is None: # single process
			start, step = 0, 1
		else:
			start, step = worker.id, worker.num_workers

		for i in range(start, self.end, step):
			for input, target in self.cutup(i):
				yield self.transform(input), self.transform(target)

	def cutup(self, i, size=256):
		'''Generator of input-target pairs of shape (@size, @size).'''

		# Load
		image = np.array(Image.open(self.images[i][0]), np.uint8)

		if os.path.exists(self.images[i][1]):
			mask = np.array(Image.open(self.images[i][1]), np.uint8)
		else:
			mask = np.zeros(image.shape[:2], np.uint8)

		# Generator
		for x in range(0, image.shape[0] - size, size):
			for y in range(0, image.shape[1] - size, size):
				subimage = image[x:x+size, y:y+size, :]
				submask = mask[x:x+size, y:y+size]
				yield subimage, submask

	@staticmethod
	def transform(array):
		'''Transforms numpy image array into torch tensor.'''
		return transforms.ToTensor()(array)

	@staticmethod
	def revert(tensor):
		'''Transforms torch tensor into PIL image.'''
		return transforms.functional.to_pil_image(tensor)


########
# Main #
########

if __name__ == '__main__':
	'''Creation of the masks'''

	# Imports
	import cv2
	import json

	# Parameters
	origin = '../resources/'
	ext = '.tif'

	# Polygons
	with open(origin + 'polygons/SolarArrayPolygons.json', 'r') as f:
		polygons = json.load(f)['polygons']

	# Table {imagename: [polygon, ...], ...}
	table = dict()

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

	# Masks creation
	for imagename, contours in table.items():
		if not os.path.exists(origin + imagename + ext):
			continue

		## Get image shape
		image = cv2.imread(origin + imagename + ext)
		mask = np.zeros(image.shape[:2], np.uint8)

		## Draw polygons interior
		cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

		## Save mask
		cv2.imwrite(origin + imagename + '_mask' + ext, mask)
