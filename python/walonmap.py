#!/usr/bin/env python

"""
Processing WalOnMap tiles

References
----------
WalOnMap - Orthorectified aerial images of Wallonia
https://geoportail.wallonie.be/walonmap
"""

#############
# Libraries #
#############

import cv2
import numpy as np

from wmts import WMTS


#############
# Constants #
#############

_WALONMAP = WMTS(
	wmts='https://geoservices.wallonie.be/arcgis/rest/services/IMAGERIE/ORTHO_2018/MapServer/WMTS',
	layer='IMAGERIE_ORTHO_2018',
	tms='default028mm',
	tm='15'
)


###########
# Classes #
###########

class Contour:
	'''OpenCV contour iterator'''

	def __init__(self, contour):
		self._contour = np.array(contour, dtype=int)
		self._len = None

	def __len__(self):
		if self._len is None:
			self._len = sum(1 for _ in self)

		return self._len

	def __iter__(self):
		for x in range(self.min[0], self.max[0] + 1):
			for y in range(self.min[1], self.max[1] + 1):
				if cv2.pointPolygonTest(self._contour, (x, y), measureDist=False) >= 0:
					yield x, y

	@property
	def min(self):
		return self._contour.min(axis=0)

	@property
	def max(self):
		return self._contour.max(axis=0)


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import json
	import os
	import torch

	import via as VIA

	from PIL import Image

	from models import MultiTaskUNet
	from dataset import to_pil, to_tensor, to_polygons

	# Arguments
	parser = argparse.ArgumentParser(description='Process WalOnMap tiles')
	parser.add_argument('-d', '--destination', default=None, help='destination of the tiles')
	parser.add_argument('-l', '--limit', default=-1, type=int, help='number of tiles to process')
	parser.add_argument('-m', '--model', default='../products/models/unet.pth', help='model')
	parser.add_argument('-n', '--name', default='', help='prefix name of the tiles')
	parser.add_argument('-o', '--output', default='../products/json/walonmap.json', help='output VIA file')
	parser.add_argument('-p', '--polygon', required=True, help='GeoJSON polygon file')
	args = parser.parse_args()

	# Contour
	with open(args.polygon, 'r') as f:
		geojson = json.load(f)

	contour = list(map(
		lambda x: _WALONMAP.wgs_to_tile(x[1], x[0]),
		geojson['coordinates'][0]
	))

	contour = Contour(contour)

	print('{} tiles in region'.format(len(contour)))

	# Model
	model = MultiTaskUNet(3, 1, R=5)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)

	model.load_state_dict(torch.load(args.model, map_location=device))

	model.eval()

	# Destination
	if args.destination is not None:
		os.makedirs(args.destination, exist_ok=True)

	# Process tiles
	via = {}

	with torch.no_grad():
		for row, col in contour:
			if args.limit == 0:
				break
			else:
				args.limit -= 1

			try:
				tile = Image.open(_WALONMAP.get_tile(row, col))
			except Exception:
				continue

			inpt = to_tensor(tile).unsqueeze(0)
			inpt = inpt.to(device)

			outpt = model(inpt).cpu()[0]
			polygons = to_polygons(to_pil(outpt))

			if polygons:
				basename = args.name + '{}_{}.jpg'.format(row, col)
				via[basename] = polygons

				if args.destination is not None:
					filename = os.path.join(args.destination, basename)
					tile.save(filename)

	# Save
	if args.destination is None:
		VIA.dump(via, args.output)
	else:
		VIA.dump(via, args.output, path=args.destination)
