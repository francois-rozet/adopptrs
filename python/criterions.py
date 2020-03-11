"""
"""

###########
# Imports #
###########


###########
# Classes #
###########

class IoU():
	'''Intersection over Union'''

	def __init__(self):
		self.smooth = 1e-3

	def __call__(self, targets, outputs):
		intersection = (targets * outputs).sum()
		union = (targets + outputs).sum()
		iou = (2 * intersection + self.smooth) / (union + self.smooth)

		return 1 - iou
