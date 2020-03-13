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
	'''Dice Loss'''

	def __init__(self):
		super().__init__()

		self.smooth = 1.

	def forward(self, targets, outputs):
		inter = (targets * outputs).sum()
		union = (targets + outputs).sum()
		iou = (2. * inter + self.smooth) / (union + self.smooth)

		return 1. - iou
