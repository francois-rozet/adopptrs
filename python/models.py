"""
"""

###########
# Imports #
###########

import torch
import torch.nn as nn


#############
# Functions #
#############

def double_conv(in_channels, out_channels, kernel_size=3):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)


#########
# Class #
#########

class UNet(nn.Module):
	"""
	PyTorch implementation of
	U-Net: Convolutional Networks for Biomedical Image Segmentation
	(Ronneberger et al., 2015)
	https://arxiv.org/abs/1505.04597
	"""
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.first = double_conv(in_channels, 64)

		self.down1 = double_conv(64, 128)
		self.down2 = double_conv(128, 256)
		self.down3 = double_conv(256, 512)

		self.maxpool = nn.MaxPool2d(2)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.up3 = double_conv(256 + 512, 256)
		self.up2 = double_conv(128 + 256, 128)
		self.up1 = double_conv(128 + 64, 64)

		self.last = nn.Sequential(
			nn.Conv2d(64, out_channels, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		# Downhill
		d1 = self.first(x)

		x = self.maxpool(d1)
		d2 = self.down1(x)

		x = self.maxpool(d2)
		d3 = self.down2(x)

		x = self.maxpool(d3)
		x = self.down3(x)

		# Uphill
		x = self.upsample(x)
		x = torch.cat([x, d3], dim=1)
		x = self.up3(x)

		x = self.upsample(x)
		x = torch.cat([x, d2], dim=1)
		x = self.up2(x)

		x = self.upsample(x)
		x = torch.cat([x, d1], dim=1)
		x = self.up1(x)

		return self.last(x)