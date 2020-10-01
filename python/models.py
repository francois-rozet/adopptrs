#!/usr/bin/env python

"""
PyTorch Models
"""

###########
# Imports #
###########

import torch
import torch.nn as nn


#########
# Class #
#########

class DoubleConv(nn.Sequential):
	'''Double convolution layer'''

	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
		super().__init__(
			nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)


class SegNet(nn.Module):
	"""
	Implementation of SegNet

	References
	----------
	SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
	(Badrinarayanan et al., 2016)
	https://arxiv.org/abs/1511.00561
	"""

	def __init__(self, in_channels, out_channels, depth=2):
		super().__init__()

		self.downs = nn.ModuleList(
			[DoubleConv(in_channels, 64)] + [
				DoubleConv(64 * (2 ** i), 128 * (2 ** i))
				for i in range(depth)
			]
		)

		self.maxpool = nn.MaxPool2d(2, return_indices=True, ceil_mode=True)
		self.upsample = nn.MaxUnpool2d(2)

		self.ups = nn.ModuleList(
			[
				DoubleConv(128 * (2 ** i), 64 * (2 ** i))
				for i in reserved(range(depth))
			]
		)

		self.last = nn.Conv2d(64, out_channels, 1)
		self.sigmoid = nn.Sigmoid()

	def head(self, x):
		x = self.last(x)

		return self.sigmoid(x)

	def forward(self, x):
		indexes = []
		shapes = []

		# Downhill
		for down in self.downs[:-1]:
			x = down(x)
			indexes.append(None)
			shapes.append(x.shape[-2:])
			x, indexes[-1] = self.maxpool(x)

		x = self.downs[-1](x)

		# Uphill
		for up in self.ups:
			x = self.upsample(x, indexes.pop())
			x = x[:, :, :shapes[-1][0], :shapes.pop()[1]]
			x = up(x)

		return self.head(x)


class MultiTaskSegNet(SegNet):
	"""
	Implementation of a multi-task SegNet

	References
	----------
	Multi-Task Learning for Segmentation of Building Footprints with Deep Neural Networks
	(Bischke et al., 2019)
	https://arxiv.org/abs/1709.05932
	"""

	def __init__(self, in_channels, out_channels, R):
		super().__init__(in_channels, out_channels)

		self.dist = nn.Conv2d(64, R * 2 + 1, 1)
		self.relu = nn.ReLU(inplace=True)

		self.last = nn.Conv2d(64 + self.dist.out_channels, out_channels, 1)
		self.sigmoid = nn.Sigmoid()

	def head(self, x):
		dist = self.dist(x)

		x = torch.cat([x, self.relu(dist)], dim=1)
		x = self.last(x)
		x = self.sigmoid(x)

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

	def __init__(self, in_channels, out_channels, depth=4):
		super().__init__()

		self.downs = nn.ModuleList(
			[DoubleConv(in_channels, 64)] + [
				DoubleConv(64 * (2 ** i), 128 * (2 ** i))
				for i in range(depth)
			]
		)

		self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

		self.ups = nn.ModuleList([
			DoubleConv((64 + 128) * (2 ** i), 64 * (2 ** i))
			for i in reversed(range(depth))
		])

		self.last = nn.Conv2d(64, out_channels, 1)
		self.sigmoid = nn.Sigmoid()

	def head(self, x):
		x = self.last(x)

		return self.sigmoid(x)

	def forward(self, x):
		features = []
		shapes = []

        # Downhill
        for down in self.downs[:-1]:
            x = down(x)
            features.append(x)
            shapes.append(x.shape[-2:])
            x = self.maxpool(x)

        x = self.downs[-1](x)

		# Uphill
		for up in self.ups:
			x = self.upsample(x)
			x = torch.cat([
				x[:, :, :shapes[-1][0], :shapes.pop()[1]],
				features.pop()
			], dim=1)
			x = up(x)

		return self.head(x)


class MultiTaskUNet(UNet):
	"""
	Implementation of a multi-task UNet
	"""

	def __init__(self, in_channels, out_channels, R):
		super().__init__(in_channels, out_channels)

		self.dist = nn.Conv2d(64, R * 2 + 1, 1)
		self.relu = nn.ReLU(inplace=True)

		self.last = nn.Conv2d(64 + self.dist.out_channels, out_channels, 1)
		self.sigmoid = nn.Sigmoid()

	def head(self, x):
		dist = self.dist(x)

		x = torch.cat([x, self.relu(dist)], dim=1)
		x = self.last(x)
		x = self.sigmoid(x)

		return (x, dist) if self.training else x
