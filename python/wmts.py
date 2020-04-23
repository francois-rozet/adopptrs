#!/usr/bin/env python

"""
Web Map Tile Service package

References
----------
OpenGIS Web Map Tile Service Implementation Standard
(Maso, Pomakis and Julia, 2010)
https://www.ogc.org/standards/wmts
"""

#############
# Libraries #
#############

import pyproj as pp

from owslib.wmts import WebMapTileService


#############
# Constants #
#############

_PIXEL_SIZE = 0.28e-3
_WGS84 = 'EPSG:4326' # World Geodetic System 84


###########
# Classes #
###########

class WMTS:
	'''Web Map Tile Service wrapper'''

	def __init__(self, wmts, layer=None, tms=None, tm=None, fmt=None, pixel_size=_PIXEL_SIZE):
		# Web Map Tile Service
		self.wmts = WebMapTileService(wmts)

		if layer is None:
			layer = next(iter(self.wmts.contents))
		self.layer = self.wmts.contents[layer]

		if tms is None:
			tms = next(iter(self.wmts.tilematrixsets))
		self.tms = self.wmts.tilematrixsets[tms]

		if tm is None:
			tm = next(iter(self.tms.tilematrix))
		self.tm = self.tms.tilematrix[tm]

		# CRS
		self.crs = pp.CRS.from_user_input(self.tms.crs)
		self.epsg = ':'.join(self.crs.to_authority())

		self._from_wgs = pp.Transformer.from_crs(_WGS84, self.crs)
		self._to_wgs = pp.Transformer.from_crs(self.crs, _WGS84)

		self.unit = self.crs.coordinate_system.axis_list[0].unit_name
		self.ucf = self.crs.coordinate_system.axis_list[0].unit_conversion_factor

		# Tile span(s)
		self.pixel_size = pixel_size
		self.scale = self.tm.scaledenominator
		self.pixel_span = self.scale * self.pixel_size * self.ucf

		self.tile_width = self.tm.tilewidth
		self.tile_height = self.tm.tileheight

		self.tile_span_x = self.tile_width * self.pixel_span
		self.tile_span_y = self.tile_height * self.pixel_span

		# Domain
		self.matrix_width = self.tm.matrixwidth
		self.matrix_height = self.tm.matrixheight

		self.x_min, self.y_max = self.tm.topleftcorner
		self.x_max = self.x_min + self.matrix_width * self.tile_span_x
		self.y_min = self.y_max - self.matrix_height * self.tile_span_y

		# Format
		self.fmt = self.layer.formats[0] if fmt is None else fmt

	def get_tile(self, row, col):
		return self.wmts.gettile(
			layer=self.layer.id,
			tilematrixset=self.tms.identifier,
			tilematrix=self.tm.identifier,
			row=row,
			column=col,
			format=self.fmt
		)

	def tile_to_xy(self, row, col):
		x = self.x_min + col * self.tile_span_x
		y = self.y_max - row * self.tile_span_y

		return x, y

	def xy_to_tile(self, x, y, integer=True):
		col = (x - self.x_min) / self.tile_span_x
		row = (self.y_max - y) / self.tile_span_y

		if integer:
			row, col = int(row), int(col)

		return row, col

	def xy_to_wgs(self, x, y):
		return self._to_wgs.transform(x, y)

	def wgs_to_xy(self, lat, lon):
		return self._from_wgs.transform(lat, lon)

	def tile_to_wgs(self, row, col):
		return self.xy_to_wgs(*self.tile_to_xy(row, col))

	def wgs_to_tile(self, lat, lon):
		return self.xy_to_tile(*self.wgs_to_xy(lat, lon))

	@property
	def domain(self):
		return self.x_min, self.y_min, self.x_max, self.y_max
