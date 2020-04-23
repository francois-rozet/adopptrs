#!/usr/bin/env python

"""
PyTorch Models
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
	'''Generic double convolution layer'''
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

class SegNet(nn.Module):
	"""
	Implementation of SegNet

	References
	----------
	SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
	(Badrinarayanan et al., 2016)
	https://arxiv.org/pdf/1511.00561.pdf
	"""

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.first = double_conv(in_channels, 64)

		self.down1 = double_conv(64, 128)
		self.down2 = double_conv(128, 128)

		self.maxpool = nn.MaxPool2d(2, return_indices=True)
		self.upsample = nn.MaxUnpool2d(2)

		self.up2 = double_conv(128, 128)
		self.up1 = double_conv(128, 64)

		self.last = nn.Sequential(
			nn.Conv2d(64, out_channels, 1),
			nn.Sigmoid()
		)

	def head(self, x):
		return self.last(x)

	def forward(self, x):
		# Downhill
		x = self.first(x)
		x, idx_1 = self.maxpool(x)

		x = self.down1(x)
		x, idx_2 = self.maxpool(x)

		x = self.down2(x)
		x, idx_3 = self.maxpool(x)

		# Uphill
		x = self.upsample(x, idx_3)
		x = self.up2(x)

		x = self.upsample(x, idx_2)
		x = self.up1(x)

		x = self.upsample(x, idx_1)

		return self.head(x)


class MultiTaskSegNet(SegNet):
	"""
	Implementation of a multi-task SegNet

	References
	----------
	Multi-Task Learning for Segmentation of Building Footprints with Deep Neural Networks
	(Bischke et al., 2019)
	https://arxiv.org/pdf/1709.05932.pdf
	"""

	def __init__(self, in_channels, out_channels, R):
		super().__init__(in_channels, out_channels)

		self.dist = nn.Conv2d(64, R * 2 + 1, 1)
		self.relu = nn.ReLU(inplace=True)

		self.last = nn.Sequential(
			nn.Conv2d(64 + self.dist.out_channels, out_channels, 1),
			nn.Sigmoid()
		)

	def head(self, x):
		dist = self.dist(x)

		x = torch.cat([x, self.relu(dist)], dim=1)
		x = self.last(x)

		return (x, dist) if self.training else x


class UNet(nn.Module):
	"""
	Implementation of U-Net

	References
	----------
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
		self.down4 = double_conv(512, 1024)

		self.maxpool = nn.MaxPool2d(2)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.up4 = double_conv(512 + 1024, 512)
		self.up3 = double_conv(256 + 512, 256)
		self.up2 = double_conv(128 + 256, 128)
		self.up1 = double_conv(128 + 64, 64)

		self.last = nn.Sequential(
			nn.Conv2d(64, out_channels, 1),
			nn.Sigmoid()
		)

	def head(self, x):
		return self.last(x)

	def forward(self, x):
		# Downhill
		d1 = self.first(x)

		x = self.maxpool(d1)
		d2 = self.down1(x)

		x = self.maxpool(d2)
		d3 = self.down2(x)

		x = self.maxpool(d3)
		d4 = self.down3(x)

		x = self.maxpool(d4)
		x = self.down4(x)

		# Uphill
		x = self.upsample(x)
		x = torch.cat([x, d4], dim=1)
		x = self.up4(x)

		x = self.upsample(x)
		x = torch.cat([x, d3], dim=1)
		x = self.up3(x)

		x = self.upsample(x)
		x = torch.cat([x, d2], dim=1)
		x = self.up2(x)

		x = self.upsample(x)
		x = torch.cat([x, d1], dim=1)
		x = self.up1(x)

		return self.head(x)


class MultiTaskUNet(UNet):
	"""
	Implementation of a multi-task UNet
	"""

	def __init__(self, in_channels, out_channels, R):
		super().__init__(in_channels, out_channels)

		self.dist = nn.Conv2d(64, R * 2 + 1, 1)
		self.relu = nn.ReLU(inplace=True)

		self.last = nn.Sequential(
			nn.Conv2d(64 + self.dist.out_channels, out_channels, 1),
			nn.Sigmoid()
		)

	def head(self, x):
		dist = self.dist(x)

		x = torch.cat([x, self.relu(dist)], dim=1)
		x = self.last(x)

		return (x, dist) if self.training else x
