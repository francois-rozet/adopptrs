"""
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
	import numpy as np
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
	parser.add_argument('-d', '--destination', default='../products/models/', help='destination of the model(s)')
	parser.add_argument('-j', '--json', default='../products/json/data.json', help='file listing image-mask pairs')
	parser.add_argument('-n', '--name', default='unet', help='name of the model')
	parser.add_argument('-o', '--output', default=None, help='standard output file')
	parser.add_argument('-p', '--path', default='../resources/', help='path to resources')
	parser.add_argument('-r', '--resume', type=int, default=1, help='epoch at which to resume')
	args = parser.parse_args()

	# Output file
	if args.output is not None:
		sys.stdout = open(args.output, 'a')

	print('-' * 10)

	# Datasets
	data = LargeDataset.load(args.json, args.path)

	random.seed(0)
	random.shuffle(data)

	trainset = RandomTurn(LargeDataset(data[:350], transform='tensor', color='jitter', shuffle=True))
	validset = LargeDataset(data[350:400], transform='tensor')

	print('Training size = {}'.format(len(trainset)))
	print('Validation size = {}'.format(len(validset)))

	# Dataloaders
	trainloader = DataLoader(trainset, batch_size=5, pin_memory=True)
	validloader = DataLoader(validset, batch_size=5, pin_memory=True)

	# Model
	model = UNet(3, 1)

	if torch.cuda.is_available():
		print('CUDA available -> Transfering to CUDA')
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
		print('CUDA unavailable')

	model.to(device)

	os.makedirs(args.destination, exist_ok=True)
	basename = os.path.join(args.destination, args.name)

	if args.resume > 1:
		modelname = '{}_{:03d}.pth'.format(basename, args.resume - 1)

		if os.path.exists(modelname):
			print('Resuming from {}'.format(modelname))
			model.load_state_dict(torch.load(modelname, map_location=device))

	# Parameters
	epochs = 100
	lr, wd = 1e-3, 1e-4

	# Criterion and optimizer
	criterion = DiceLoss(smooth=1.)
	optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

	# Training
	best_loss = 1.
	epochs = range(args.resume, args.resume + epochs)

	for epoch in epochs:
		if args.output is not None:
			sys.stdout.close()
			sys.stdout = open(args.output, 'a')

		print('-' * 10)
		print('Epoch {}'.format(epoch))

		start = time.time()

		## Training set
		losses = train_epoch(model, trainloader, criterion, optimizer)
		mean, std = np.mean(losses), np.std(losses)
		print('Training loss = {} +- {}'.format(mean, std))

		## Validation set
		losses = eval(model, validloader, [criterion])
		mean, std = np.mean(losses), np.std(losses)
		print('Validation loss = {} +- {}'.format(mean, std))

		elapsed = time.time() - start

		print('{:.0f}m{:.0f}s elapsed'.format(elapsed // 60, elapsed % 60))

		if mean < best_loss or epoch == epochs[-1]:
			best_loss = mean

			modelname = '{}_{:03d}.pth'.format(basename, epoch)

			print('Saving {}'.format(modelname))

			torch.save(model.state_dict(), modelname)
