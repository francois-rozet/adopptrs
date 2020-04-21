"""
"""

#############
# Constants #
#############

_TOL = 1e-3


#############
# Functions #
#############

def parse(imagename):
	return tuple(
		map(
			int,
			imagename.split('.')[0].split('_')[-2:]
		)
	)


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import cv2
	import csv
	import numpy as np
	import os

	import via as VIA
	from walonmap import _WALONMAP as wm

	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default='../products/json/walonmap.json', help='input VIA file')
	parser.add_argument('-o', '--output', default='../products/csv/summary.csv', help='output csv file')
	parser.add_argument('-t', '--threshold', default=10., type=float, help='area threshold')
	args = parser.parse_args()

	# VIA
	via = VIA.load(args.input)

	# Panels
	os.makedirs(os.path.dirname(args.output), exist_ok=True)

	with open(args.output, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['latitude', 'longitude', 'area', 'azimuth'])

		for imagename, polygons in via.items():
			row, col = parse(imagename)

			for polygon in polygons:
				panel = np.array([
					wm.tile_to_xy(row + y / wm.tile_height, col + x / wm.tile_width)
					for x, y in polygon
				])
				x, y = panel.mean(axis=0)
				panel = (panel / _TOL).astype(int)

				area = cv2.contourArea(panel) * (_TOL ** 2)
				_, _, angle = cv2.minAreaRect(panel)

				## Threshold
				if area > args.threshold:
					lat, lon = wm.xy_to_wgs(x, y)
					azimuth = 180 + angle if angle > -45 else 270 + angle

					writer.writerow([
						round(lat, 6),
						round(lon, 6),
						round(area, 2),
						round(azimuth, 2)
					])
