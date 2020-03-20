"""
"""

###########
# Imports #
###########

import torch.nn as nn


###########
# Classes #
###########

class DiceLoss(nn.Module):
	'''Dice Loss (IoU, F-score, ...).'''

	def __init__(self, smooth=1.):
		super().__init__()

		self.smooth = smooth

	def forward(self, targets, outputs):
		inter = (targets * outputs).sum()
		union = targets.sum() + outputs.sum()
		iou = (2. * inter + self.smooth) / (union + self.smooth)

		return 1. - iou


class TP(nn.Module):
	'''True Positive.'''

	def __init__(self):
		super().__init__()

	def forward(self, targets, outputs):
		return (targets * outputs).sum()


class TN(nn.Module):
	'''True Negative.'''

	def __init__(self):
		super().__init__()

	def forward(self, targets, outputs):
		return ((1 - targets) * (1 - outputs)).sum()


class FP(nn.Module):
	'''False Positive.'''

	def __init__(self):
		super().__init__()

	def forward(self, targets, outputs):
		return ((1 - targets) * outputs).sum()


class FN(nn.Module):
	'''False Negative.'''

	def __init__(self):
		super().__init__()

	def forward(self, targets, outputs):
		return (targets * (1 - outputs)).sum()
