#!/usr/bin/env python

"""
Training and validation of models
"""

###########
# Imports #
###########

import time
import torch


#############
# Functions #
#############

def train_epoch(model, loader, criterion, optimizer):
	'''Trains a model for one epoch.'''
	model.train()
	device = next(model.parameters()).device
	losses = []

	for inputs, targets in loader:
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = model(inputs)

		loss = criterion(outputs, targets)
		losses.append(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return losses


def eval(model, loader, metrics):
	'''Evaluates metrics on a model.'''
	model.eval()
	device = next(model.parameters()).device
	values = []

	with torch.no_grad():
		for inputs, targets in loader:
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = model(inputs)

			values.append(
				[metric(outputs, targets).item() for metric in metrics]
			)

	return values


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import csv
	import json
	import numpy as np
	import os
	import random
	import sys
	import via as VIA

	from torch.utils.data import DataLoader
	from torch.optim import Adam, SGD

	from dataset import VIADataset, ColorJitter, RandomFilter, RandomTranspose, Scale, ToTensor
	from models import UNet, SegNet
	from criterions import DiceLoss

	# Arguments
	parser = argparse.ArgumentParser(description='Train and validate a model')
	parser.add_argument('-d', '--destination', default='../products/models/', help='destination of the model(s)')
	parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
	parser.add_argument('-i', '--input', default='../products/json/california.json', help='input VIA file')
	parser.add_argument('-m', '--model', default='unet', help='model')
	parser.add_argument('-n', '--name', default=None, help='name of the model')
	parser.add_argument('-o', '--output', default=None, help='standard output file')
	parser.add_argument('-p', '--path', default='../resources/california/', help='path to resources')
	parser.add_argument('-r', '--resume', type=int, default=1, help='epoch at which to resume')
	parser.add_argument('-s', '--split', type=int, nargs=2, default=(350, 400), help='train-valid-test splitting indexes')
	parser.add_argument('-stat', default='../products/csv/statistics.csv', help='statistics file')
	parser.add_argument('-optim', default='adam', help='optimizer')
	parser.add_argument('-lrate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('-wdecay', type=float, default=1e-4, help='weight decay')
	parser.add_argument('-momentum', type=float, default=0.9, help='momentum')
	args = parser.parse_args()

	# Output file
	if args.output is not None:
		sys.stdout = open(args.output, 'a')

	print('-' * 10)

	# Datasets
	via = VIA.load(args.input)

	keys = sorted(list(via.keys()))

	random.seed(0)
	random.shuffle(keys)

	train_via = {key: via[key] for key in keys[:args.split[0]]}
	valid_via = {key: via[key] for key in keys[args.split[0]:args.split[1]]}

	trainset = ToTensor(RandomTranspose(RandomFilter(ColorJitter(VIADataset(train_via, args.path, shuffle=True)))))
	validset = ToTensor(VIADataset(valid_via, args.path))

	"""Scaling the images for WalOnMap
	trainset = ToTensor(RandomTranspose(RandomFilter(ColorJitter(Scale(VIADataset(train_via, args.path, size=128, shuffle=True), 2)))))
	validset = ToTensor(Scale(VIADataset(valid_via, size=128, args.path), 2))
	"""

	print('Training size = {}'.format(len(trainset)))
	print('Validation size = {}'.format(len(validset)))

	# Dataloaders
	trainloader = DataLoader(trainset, batch_size=5, pin_memory=True)
	validloader = DataLoader(validset, batch_size=5, pin_memory=True)

	# Model
	if args.model == 'unet':
		model = UNet(3, 1)
	elif args.model == 'segnet':
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

	os.makedirs(args.destination, exist_ok=True)
	basename = os.path.join(
		args.destination,
		args.model if args.name is None else args.name
	)

	if args.resume > 1:
		modelname = '{}_{:03d}.pth'.format(basename, args.resume - 1)

		if os.path.exists(modelname):
			print('Resuming from {}'.format(modelname))
			model.load_state_dict(torch.load(modelname, map_location=device))

	# Statistics
	os.makedirs(os.path.dirname(args.stat), exist_ok=True)

	if not os.path.exists(args.stat):
		with open(args.stat, 'w', newline='') as f:
			csv.writer(f).writerow([
				'model',
				'epoch',
				'train_loss_mean',
				'train_loss_std',
				'train_loss_first',
				'train_loss_second',
				'train_loss_third',
				'valid_loss_mean',
				'valid_loss_std',
				'valid_loss_first',
				'valid_loss_second',
				'valid_loss_third'
			])

	# Criterion
	criterion = DiceLoss(smooth=1.)

	# Optimizer
	if args.optim == 'adam':
		optimizer = Adam(
			model.parameters(),
			lr=args.lrate,
			weight_decay=args.wdecay
		)
	elif args.optim == 'sgd':
		optimizer = SGD(
			model.parameters(),
			lr=args.lrate,
			weight_decay=args.wdecay,
			momentum=args.momentum
		)
	else:
		raise ValueError('unknown optimizer {}'.format(args.optim))

	# Training
	best_loss = 1.
	epochs = range(args.resume, args.resume + args.epochs)

	for epoch in epochs:
		if args.output is not None:
			sys.stdout.close()
			sys.stdout = open(args.output, 'a')

		print('-' * 10)
		print('Epoch {}'.format(epoch))

		start = time.time()

		## Training set
		train_losses = train_epoch(model, trainloader, criterion, optimizer)

		## Validation set
		valid_losses = eval(model, validloader, [criterion])

		elapsed = time.time() - start

		print('{:.0f}m{:.0f}s elapsed'.format(elapsed // 60, elapsed % 60))

		## Statistics
		train_losses = np.array(train_losses)
		valid_losses = np.array(valid_losses)

		train_mean = train_losses.mean()
		valid_mean = valid_losses.mean()

		print('Training loss = {}'.format(train_mean))
		print('Validation loss = {}'.format(valid_mean))

		with open(args.stat, 'a', newline='') as f:
			csv.writer(f).writerow([
				args.name,
				epoch,
				np.mean(train_losses),
				np.std(train_losses),
				np.quantile(train_losses, 0.25),
				np.quantile(train_losses, 0.5),
				np.quantile(train_losses, 0.75),
				np.mean(valid_losses),
				np.std(valid_losses),
				np.quantile(valid_losses, 0.25),
				np.quantile(valid_losses, 0.5),
				np.quantile(valid_losses, 0.75),
			])

		if valid_mean < best_loss or epoch == epochs[-1]:
			best_loss = valid_mean

			modelname = '{}_{:03d}.pth'.format(basename, epoch)

			print('Saving {}'.format(modelname))

			torch.save(model.state_dict(), modelname)
