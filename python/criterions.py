#!/usr/bin/env python

"""
PyTorch loss functions and metrics
"""

###########
# Imports #
###########

import torch
import torch.nn as nn

from scipy.ndimage.morphology import distance_transform_edt as dt

###########
# Classes #
###########

class MultiTaskLoss(nn.Module):
	"""
	Distance loss and Dice loss sum

	References
	----------
	Multi-Task Learning for Segmentation of Building Footprints with Deep Neural Networks
	(Bischke et al., 2019)
	https://arxiv.org/pdf/1709.05932.pdf
	"""

	def __init__(self, smooth=1., R=5):
		super().__init__()

		self.dice = DiceLoss(smooth=smooth)
		self.cross = nn.CrossEntropyLoss()

		self.R = R

	def forward(self, outputs, targets):
		dists = self.transform(targets)
		return self.dice(outputs[0], targets) + self.cross(outputs[1], dists)

	def transform(self, targets):
		'''Transforms targets into distances.'''
		dists = targets.cpu().squeeze(dim=1).numpy()

		for i in range(len(dists)):
			dists[i, ...] = dt(dists[i, ...]) - dt(1 - dists[i, ...])

		return torch.clamp(
			torch.tensor(dists, dtype=int, device=targets.device),
			min=-self.R,
			max=self.R
		) + self.R


class DiceLoss(nn.Module):
	'''Dice Loss (F-score, ...).'''

	def __init__(self, smooth=1.):
		super().__init__()

		self.smooth = smooth

	def forward(self, outputs, targets):
		inter = (outputs * targets).sum()
		union = outputs.sum() + targets.sum()
		iou = (2. * inter + self.smooth) / (union + self.smooth)

		return 1. - iou


class TP(nn.Module):
	'''True Positive.'''

	def __init__(self, threshold=0.5):
		super().__init__()

		self.threshold = threshold

	def forward(self, outputs, targets):
		return ((outputs > self.threshold) * targets).sum()


class TN(nn.Module):
	'''True Negative.'''

	def __init__(self, threshold=0.5):
		super().__init__()

		self.threshold = threshold

	def forward(self, outputs, targets):
		return ((outputs < self.threshold) * (1 - targets)).sum()


class FP(nn.Module):
	'''False Positive.'''

	def __init__(self, threshold=0.5):
		super().__init__()

		self.threshold = threshold

	def forward(self, outputs, targets):
		return ((outputs > self.threshold) * (1 - targets)).sum()


class FN(nn.Module):
	'''False Negative.'''

	def __init__(self, threshold=0.5):
		super().__init__()

		self.threshold = threshold

	def forward(self, outputs, targets):
		return ((outputs < self.threshold) * targets).sum()
