#!/usr/bin/env python

"""
Evaluation of models
"""

###########
# Imports #
###########

import cv2
import numpy as np


#############
# Functions #
#############

def bounding(contour):
	'''Computes the bounding box of an OpenCV contour.'''
	y_min, x_min = np.amin(contour[:, 0, :], axis=0)
	y_max, x_max = np.amax(contour[:, 0, :], axis=0)

	return x_min, y_min, x_max, y_max


def intersection(a, b):
	'''Computes the intersection box of two boxes.'''
	x_min = max(a[0], b[0])
	y_min = max(a[1], b[1])
	x_max = min(a[2], b[2])
	y_max = min(a[3], b[3])

	if x_max > x_min and y_max > y_min:
		return x_min, y_min, x_max, y_max
	return None


def surface(contour):
	'''Estimates the surface of an OpenCV contour.'''
	return cv2.arcLength(contour, True) / 2 + cv2.contourArea(contour)


def safe_divide(a, b, default=1):
	'''Guess what ? It divides safely.'''
	return np.divide(a, b, out=np.zeros_like(a) + default, where=b != 0)


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import os
	import random
	import sys
	import torch
	import via as VIA

	from dataset import VIADataset, ToTensor, to_pil, to_contours
	from models import UNet, SegNet, MultiTaskUNet, MultiTaskSegNet
	from criterions import TP, TN, FP, FN

	# Arguments
	parser = argparse.ArgumentParser(description='Evaluate a model')
	parser.add_argument('-f', '--fold', type=int, default=0, help='the fold')
	parser.add_argument('-i', '--input', default='../products/json/california.json', help='input VIA file')
	parser.add_argument('-k', type=int, default=5, help='the number of folds')
	parser.add_argument('-m', '--model', default='unet', choices=['unet', 'segnet'], help='network schema')
	parser.add_argument('-multitask', default=False, action='store_true', help='multi-task network')
	parser.add_argument('-n', '--network', default=None, help='network file')
	parser.add_argument('-o', '--output', default=None, help='standard output file')
	parser.add_argument('-p', '--path', default='../resources/california/', help='path to resources')
	parser.add_argument('-min', type=int, default=64, help='minimal number of pixels')
	args = parser.parse_args()

	# Output file
	if args.output is not None:
		os.makedirs(os.path.dirname(args.output), exist_ok=True)
		sys.stdout = open(args.output, 'a')

	print('-' * 10)

	# Datasets
	via = VIA.load(args.input)

	keys = sorted(list(via.keys()))

	random.seed(0) # reproductability
	random.shuffle(keys)

	if (args.k > 0):
		valid_via = {key: via[key] for i, key in enumerate(keys) if (i % args.k) == args.fold}
	else:
		valid_via = {}

	validset = ToTensor(VIADataset(valid_via, args.path, size=512, full=True))

	print('Validation size = {}'.format(len(validset)))

	# Model
	if args.model == 'unet':
		if args.multitask:
			model = MultiTaskUNet(3, 1, R=5)
		else:
			model = UNet(3, 1)
	elif args.model == 'segnet':
		if args.multitask:
			model = MultiTaskSegNet(3, 1, R=5)
		else:
			model = SegNet(3, 1)
	else:
		raise ValueError('unknown model {}'.format(args.model))

	if torch.cuda.is_available():
		print('CUDA available -> Transfering to CUDA')
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
		print('CUDA unavailable')

	model = model.to(device)
	model.load_state_dict(torch.load(args.network, map_location=device))

	# Thresholds
	thresholds = [0]
	thresholds.extend(map(lambda x: 10 ** x, range(-9, 0))) # [1e-9, 1e-8, ...]
	thresholds.append(0.5)
	thresholds.extend(map(lambda x: 1 - 10 ** x, range(-1, -10, -1)))  # [1 - 1e-1, 1 - 1e-2, ...]
	thresholds.append(1)

	# TP, FP, FN
	contour_wise = np.zeros((len(thresholds), 5))
	pixel_wise = np.zeros((len(thresholds), 5))

	# Evaluation
	model.eval()

	with torch.no_grad():
		for inpt, target in validset:
			if target.sum().item() < 1:
				continue

			inpt = inpt.unsqueeze(0).to(device)
			outpt = model(inpt).cpu()[0]

			## Target's contours
			target_ctns = [
				[bounding(c), surface(c), False]
				for c in to_contours(np.array(to_pil(target)))
			]

			for i, t in enumerate(thresholds):
				## Output's contours
				thresh = (outpt > t).float()

				output_ctns = [
					[bounding(c), surface(c), False]
					for c in to_contours(np.array(to_pil(thresh)))
				]

				inter = target * thresh

				## Matching
				for j in range(len(target_ctns)):
					target_ctns[j][2] = False

					for k in range(len(output_ctns)):
						box = intersection(target_ctns[j][0], output_ctns[k][0])

						if box is not None:
							area = inter[0, box[0]:(box[2] + 1), box[1]:(box[3] + 1)].sum().item()

							if area > 0: # it's a match !
								target_ctns[j][2] = True
								output_ctns[k][2] = True

				## Confusion metrics
				for box, area, matched in target_ctns:
					if matched:
						contour_wise[i, 0] += 1
						pixel_wise[i, 2] += area
					elif area > args.min: # too small
						contour_wise[i, 2] += 1

				for box, area, matched in output_ctns:
					if matched:
						pixel_wise[i, 1] += area
					elif area > args.min: # too small
						contour_wise[i, 1] += 1

				common = inter.sum().item()
				pixel_wise[i, 0] += common
				pixel_wise[i, 1] -= common
				pixel_wise[i, 2] -= common

	# Outputs

	""" N.B.
	The contour-wise precision at very low thresholds and pixel-wise
	recall at very high thresholds should be set to zero by hand.
	"""

	contour_wise[:, 3] = safe_divide(contour_wise[:, 0], contour_wise[:, 0] + contour_wise[:, 1])
	contour_wise[:, 4] = safe_divide(contour_wise[:, 0], contour_wise[:, 0] + contour_wise[:, 2])

	print('Contour wise :', np.array2string(contour_wise, separator=', '), sep='\n')

	pixel_wise[:, 3] = safe_divide(pixel_wise[:, 0], pixel_wise[:, 0] + pixel_wise[:, 1])
	pixel_wise[:, 4] = safe_divide(pixel_wise[:, 0], pixel_wise[:, 0] + pixel_wise[:, 2])

	print('Pixel wise :', np.array2string(pixel_wise, separator=', '), sep='\n')
