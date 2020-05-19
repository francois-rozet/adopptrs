#!/usr/bin/env python

"""
Training of models
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


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import csv
	import numpy as np
	import os
	import random
	import sys
	import via as VIA

	from torch.utils.data import DataLoader
	from torch.optim import Adam, SGD

	from dataset import VIADataset, ColorJitter, RandomFilter, RandomTranspose, Scale, ToTensor
	from models import UNet, SegNet, MultiTaskUNet, MultiTaskSegNet
	from criterions import DiceLoss, MultiTaskLoss, TP, TN, FP, FN

	# Arguments
	parser = argparse.ArgumentParser(description='Train a model')
	parser.add_argument('-d', '--destination', default='../products/models/', help='destination of the network file(s)')
	parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
	parser.add_argument('-f', '--fold', type=int, default=0, help='the fold')
	parser.add_argument('-i', '--input', default='../products/json/california.json', help='input VIA file')
	parser.add_argument('-k', type=int, default=5, help='the number of folds')
	parser.add_argument('-m', '--model', default='unet', choices=['unet', 'segnet'], help='network schema')
	parser.add_argument('-multitask', default=False, action='store_true', help='multi-task network')
	parser.add_argument('-n', '--name', default=None, help='name of the network')
	parser.add_argument('-o', '--output', default=None, help='standard output file')
	parser.add_argument('-p', '--path', default='../resources/california/', help='path to resources')
	parser.add_argument('-r', '--resume', type=int, default=1, help='epoch at which to resume')
	parser.add_argument('-s', '--stat', default='../products/csv/statistics.csv', help='convergence statistics file')
	parser.add_argument('-scale', type=int, default=1, help='scale of the images')
	parser.add_argument('-batch', type=int, default=5, help='batch size')
	parser.add_argument('-optim', default='adam', choices=['adam', 'sgd'], help='optimizer')
	parser.add_argument('-lrate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('-wdecay', type=float, default=1e-4, help='weight decay')
	parser.add_argument('-momentum', type=float, default=0.9, help='momentum of SGD')
	parser.add_argument('-special', default=False, action='store_true', help='special mode')
	args = parser.parse_args()

	# Output file
	if args.output is not None:
		if os.path.dirname(args.output):
			os.makedirs(os.path.dirname(args.output), exist_ok=True)
		sys.stdout = open(args.output, 'a')

	print('-' * 10)

	# Datasets
	via = VIA.load(args.input)

	keys = sorted(list(via.keys()))

	random.seed(0) # reproductability
	random.shuffle(keys)

	if (args.k > 0):
		train_via = {key: via[key] for i, key in enumerate(keys) if (i % args.k) != args.fold}
	else:
		train_via = via

	if args.special:
		trainset = VIADataset(train_via, args.path, shuffle=True, size=None)
		trainset = RandomTranspose(trainset)
	else:
		trainset = VIADataset(train_via, args.path, shuffle=True, alt=1)
		trainset = Scale(trainset, args.scale) if args.scale > 1 else trainset
		trainset = RandomTranspose(RandomFilter(ColorJitter(trainset)))

	trainset = ToTensor(trainset)

	print('Training size = {}'.format(len(trainset)))

	# Dataloaders
	trainloader = DataLoader(trainset, batch_size=args.batch, pin_memory=True)

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
		else:
			args.resume = 1
	else:
		args.resume = 1

	# Convergence statistics
	if os.path.dirname(args.stat):
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
				'train_loss_third'
			])

	# Criterion
	if args.multitask:
		train_criterion = MultiTaskLoss(smooth=1., R=5)
	else:
		train_criterion = DiceLoss(smooth=1.)

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
	epochs = range(args.resume, args.resume + args.epochs)

	for epoch in epochs:
		if args.output is not None:
			sys.stdout.close()
			sys.stdout = open(args.output, 'a')

		print('-' * 10)
		print('Epoch {}'.format(epoch))

		start = time.time()

		## Training set
		train_losses = train_epoch(model, trainloader, train_criterion, optimizer)

		elapsed = time.time() - start

		print('{:.0f}m{:.0f}s elapsed'.format(elapsed // 60, elapsed % 60))

		## Statistics
		train_losses = np.array(train_losses)

		train_mean = train_losses.mean()

		print('Training loss = {}'.format(train_mean))

		with open(args.stat, 'a', newline='') as f:
			csv.writer(f).writerow([
				args.name,
				epoch,
				np.mean(train_losses),
				np.std(train_losses),
				np.quantile(train_losses, 0.25),
				np.quantile(train_losses, 0.5),
				np.quantile(train_losses, 0.75)
			])

		## Saving last
		if epoch == epochs[-1]:
			modelname = '{}_{:03d}.pth'.format(basename, epoch)
			print('Saving {}'.format(modelname))
			torch.save(model.state_dict(), modelname)
