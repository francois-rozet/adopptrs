# Imports
import sys

import os
import via as VIA

from PIL import Image
from walonmap import _WALONMAP as wm
from summarize import parse

# Arguments
parser = argparse.ArgumentParser(description='Download images from WalOnMap')
parser.add_argument('-d', '--destination', default='../products/liege/', help='destination of the tiles')
parser.add_argument('-i', '--input', default='../resources/walonmap/via_liege_city.json', help='input VIA file')
args = parser.parse_args()

# Destination
os.makedirs(args.destination, exist_ok=True)

# Download
via = VIA.load(args.input)

for imagename in via:
	row, col = parse(imagename)
	img = Image.open(wm.get_tile(row, col))
	img.save(os.path.join(args.destination, imagename))
