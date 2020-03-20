"""
"""

###########
# Imports #
###########

import numpy as np
import time
import torch


#############
# Functions #
#############

def train_epoch(model, loader, criterion, optimizer):
	model.train()
	losses = []

	for inputs, targets in loader:
		inputs = inputs.cuda()
		targets = targets.cuda()
		outputs = model(inputs)

		loss = criterion(targets, outputs)
		losses.append(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return losses


def eval(model, loader, metrics):
	model.eval()
	values = []

	with torch.no_grad():
		for inputs, targets in loader:
			inputs = inputs.cuda()
			targets = targets.cuda()
			outputs = model(inputs)

			values.append(
				[metric(targets, outputs).item() for metric in metrics]
			)

	return values


def train(model, loaders, criterion, optimizer, epochs, shuffle=True):
	for epoch in epochs:
		print('-' * 10)
		print('Epoch {}'.format(epoch))

		start = time.time()

		# Training
		losses = train_epoch(model, loaders[0], criterion, optimizer)
		mean, std = np.mean(losses), np.std(losses)
		print('Training loss = {} +- {}'.format(mean, std))

		# Validation
		losses = eval(model, loaders[1], [criterion])
		mean, std = np.mean(losses), np.std(losses)
		print('Validation loss = {} +- {}'.format(mean, std))

		elapsed = time.time() - start

		print('{:.0f}m{:.0f}s elapsed'.format(elapsed // 60, elapsed % 60))

		yield epoch, mean

		# Shuffling
		if shuffle:
			loaders[0].dataset.shuffle()


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import json
	import os
	import random
	import sys

	from torch.utils.data import DataLoader
	from torch.optim import Adam

	from dataset import LargeDataset, RandomTurn
	from models import UNet
	from criterions import DiceLoss

	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--log', default=None)
	parser.add_argument('--name', default='unet')
	parser.add_argument('--resume', type=int, default=1)
	args = parser.parse_args()

	# Log file
	if args.log is not None:
		sys.stdout = open(args.log, 'a')

	# Datasets
	with open('products/json/data.json', 'r') as f:
		data = [
			('resources/' + x, 'resources/' + y, box)
			for x, y, box in json.load(f)
		]

	random.seed(0)
	random.shuffle(data)

	trainset = RandomTurn(LargeDataset(data[:350], transform='tensor', color='jitter'))
	validset = LargeDataset(data[350:400], transform='tensor')

	# Dataloaders
	trainloader = DataLoader(trainset, batch_size=5)
	validloader = DataLoader(validset, batch_size=5)

	# Model
	model = UNet(3, 1).cuda()

	os.makedirs('products/models/', exist_ok=True)
	basename = 'products/models/' + args.name

	if args.resume > 1:
		modelname = '{}_{:03d}.pth'.format(basename, args.resume - 1)
		model.load_state_dict(torch.load(modelname))

	# Parameters
	epochs = 100
	lr, wd = 1e-3, 1e-4

	# Criterion and optimizer
	criterion = DiceLoss(smooth=1.)
	optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

	# Training
	best_loss = 1.
	epochs = range(args.resume, args.resume + epochs)

	gen = train(
		model,
		(trainloader, validloader),
		criterion,
		optimizer,
		epochs
	)

	for epoch, loss in gen:
		if loss < best_loss or epoch == epochs[-1]:
			best_loss = loss

			modelname = '{}_{:03d}.pth'.format(basename, epoch)

			print('Saving {}'.format(modelname))

			torch.save(model.state_dict(), modelname)

		if args.log is not None:
			sys.stdout.close()
			sys.stdout = open(args.log, 'a')
