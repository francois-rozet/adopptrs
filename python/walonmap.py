"""
"""

#############
# Libraries #
#############

import cv2
import numpy as np
import pyproj as pp

from owslib.wmts import WebMapTileService
from PIL import Image


#############
# Constants #
#############

_WMTS = WebMapTileService('https://geoservices.wallonie.be/arcgis/rest/services/IMAGERIE/ORTHO_LAST/MapServer/WMTS')
_LAYER = 'IMAGERIE_ORTHO_LAST'
_TILE_MATRIX_SET = 'default028mm'
_TILE_MATRIX = '15'
_FORMAT = 'image/jpg'

_WIDTH = 512
_HEIGHT = 512

_WGS84 = 'EPSG:4326' # World Geodetic System 84
_LAMBERT = 'EPSG:31370' # Belgian Lambert 72

_TO_LAMBERT = pp.Transformer.from_crs(_WGS84, _LAMBERT)
_TO_WGS84 = pp.Transformer.from_crs(_LAMBERT, _WGS84)

_X_MIN = 42000
_X_MAX = 296000
_Y_MIN = 20000
_Y_MAX = 168000

_ROW_MIN = 609074
_ROW_MAX = 611259
_COL_MIN = 530235
_COL_MAX = 533985

_TILE_WIDTH = (_X_MAX - _X_MIN) / (_COL_MAX - _COL_MIN + 1)
_TILE_HEIGHT = (_Y_MAX - _Y_MIN) / (_ROW_MAX - _ROW_MIN + 1)


#############
# Functions #
#############

def xy_to_wgs(x, y):
	return _TO_WGS84.transform(x, y)


def wgs_to_xy(lat, lon):
	return _TO_LAMBERT.transform(lat, lon)


def tile_to_xy(row, col):
	x = _X_MIN + (col - _COL_MIN) * _TILE_WIDTH
	y = _Y_MAX - (row - _ROW_MIN) * _TILE_HEIGHT

	return x, y


def xy_to_tile(x, y, integer=True):
	col = _COL_MIN + (x - _X_MIN) / _TILE_WIDTH
	row = _ROW_MIN + (_Y_MAX - y) / _TILE_HEIGHT

	if integer:
		row, col = int(row), int(col)

	return row, col


def get_tile(row, col):
	return Image.open(_WMTS.gettile(
		layer=_LAYER,
		tilematrixset=_TILE_MATRIX_SET,
		tilematrix=_TILE_MATRIX,
		row=row,
		column=col,
		format=_FORMAT
	))


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

	from models import UNet
	from dataset import to_pil, to_tensor, to_polygons

	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--destination', default=None, help='destination of the tiles')
	parser.add_argument('-l', '--limit', default=-1, type=int, help='number of tiles to process')
	parser.add_argument('-m', '--model', default='../products/models/unet.pth', help='model')
	parser.add_argument('-n', '--name', default='', help='prefix name of the tiles')
	parser.add_argument('-o', '--output', default='../products/json/walonmap.json', help='output VIA file')
	parser.add_argument('-p', '--polygon', default=None, help='GeoJSON polygon file')
	args = parser.parse_args()

	# Contour
	if args.polygon is None:
		contour = [
			(_ROW_MIN, _COL_MIN),
			(_ROW_MIN, _COL_MAX),
			(_ROW_MAX, _COL_MAX),
			(_ROW_MAX, _COL_MIN)
		]
	else:
		with open(args.polygon, 'r') as f:
			geojson = json.load(f)

		contour = list(map(
			lambda x: xy_to_tile(*wgs_to_xy(*reversed(x))),
			geojson['coordinates'][0]
		))

	contour = Contour(contour)

	print('{} tiles in region'.format(len(contour)))

	# Model
	model = UNet(3, 1)
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
				tile = get_tile(row, col)
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
