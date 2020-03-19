"""
"""

###########
# Imports #
###########

import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms


###########
# Classes #
###########

class Augmenter(data.IterableDataset):
	'''Iterable image augmenter.'''

	def __init__(self, image, mask):
		self.image = image
		self.mask = mask

	def __len__(self):
		return len(Augmenter.functions())

	def __iter__(self):
		for f, g in Augmenter.functions():
			if f is None:
				newimage = self.image
			else:
				newimage = f(self.image).astype(np.uint8)

			if g is None:
				newmask = self.mask
			else:
				newmask = g(self.mask).astype(np.uint8)

			yield newimage, newmask

	@staticmethod
	def functions():
		return [
			# Flip
			(np.flipud, np.flipud),
			(np.fliplr, np.fliplr),
			# Rotation
			(lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 1)),
			(lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 2)),
			(lambda x: np.rot90(x, 3), lambda x: np.rot90(x, 3)),
			# Brightness
			(lambda x: 0.8 * x, None),
			(lambda x: 0.9 * x, None),
			(lambda x: 0.8 * x + 0.2 * 255, None),
			(lambda x: 0.9 * x + 0.1 * 255, None)
		]


class LargeDataset(data.IterableDataset):
	'''Iterable dataset for large images.'''

	def __init__(self, data, origin=None):
		self.data = data

		if origin is not None:
			for x in self.data:
				x[0] = origin + x[0]
				x[1] = origin + x[1]

	def __len__(self):
		m = len(Augmenter(None, None)) + 1

		n = 0
		for _, _, boxes in self.data:
			n += len(boxes) * m

		return n

	def __iter__(self):
		# Multiprocessing
		worker = data.get_worker_info()

		if worker is None: # single process
			start, step = 0, 1
		else:
			start, step = worker.id, worker.num_workers

		# Iterations
		for i in range(start, len(self.data), step):
			imagename, maskname, boxes = self.data[i]

			image = np.array(Image.open(imagename).convert('RGB'))
			mask = np.array(Image.open(maskname))

			for xmin, ymin, xmax, ymax in boxes:
				subimage = image[xmin:xmax, ymin:ymax, :]
				submask = mask[xmin:xmax, ymin:ymax]

				yield self.transform(subimage), self.transform(submask)

				for newimage, newmask in Augmenter(subimage, submask):
					yield self.transform(newimage), self.transform(newmask)

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
	'''Creation of masks and computation of subimage locations.'''

	# Imports
	import cv2
	import json
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
		boxes = list()

		for x in range(0, mask.shape[0] - size, step):
			for y in range(0, mask.shape[1] - size, step):
				if np.any(mask[x:x+size, y:y+size]):
					boxes.append((x, y, x+size, y+size))

		data.append((
			imagename + ext,
			maskname + ext,
			boxes
		))

	# mkdir -p destination
	os.makedirs(destination, exist_ok=True)

	with open(destination + 'data.json', 'w') as f:
		json.dump(data, f)
