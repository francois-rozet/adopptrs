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

	def forward(self, outputs, targets):
		inter = (outputs * targets).sum()
		union = outputs.sum() + targets.sum()
		iou = (2. * inter + self.smooth) / (union + self.smooth)

		return 1. - iou


class TP(nn.Module):
	'''True Positive.'''

	def __init__(self):
		super().__init__()

	def forward(self, outputs, targets):
		return (outputs * targets).sum()


class TN(nn.Module):
	'''True Negative.'''

	def __init__(self):
		super().__init__()

	def forward(self, outputs, targets):
		return ((1 - outputs) * (1 - targets)).sum()


class FP(nn.Module):
	'''False Positive.'''

	def __init__(self):
		super().__init__()

	def forward(self, outputs, targets):
		return (outputs * (1 - targets)).sum()


class FN(nn.Module):
	'''False Negative.'''

	def __init__(self):
		super().__init__()

	def forward(self, outputs, targets):
		return ((1 - outputs) * targets).sum()
