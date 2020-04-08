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
	import csv
	import json
	import numpy as np
	import os
	import random
	import sys
	import via as VIA

	from torch.utils.data import DataLoader
	from torch.optim import Adam

	from dataset import VIADataset, ColorJitter, RandomFilter, RandomTranspose, ToTensor
	from models import UNet
	from criterions import DiceLoss

	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--destination', default='../products/models/', help='destination of the model(s)')
	parser.add_argument('-i', '--input', default='../products/json/california.json', help='input VIA file')
	parser.add_argument('-n', '--name', default='unet', help='name of the model')
	parser.add_argument('-o', '--output', default=None, help='standard output file')
	parser.add_argument('-p', '--path', default='../resources/california/', help='path to resources')
	parser.add_argument('-r', '--resume', type=int, default=1, help='epoch at which to resume')
	parser.add_argument('-s', '--stat', default='../products/csv/statistics.csv', help='statistics file')
	args = parser.parse_args()

	# Output file
	if args.output is not None:
		sys.stdout = open(args.output, 'a')

	print('-' * 10)

	# Datasets
	with open('../products/json/california.json', 'r') as f:
		via = VIA.deformat(json.load(f))

	keys = sorted(list(via.keys()))

	random.seed(0)
	random.shuffle(keys)

	train_via = {key: via[key] for key in keys[:350]}
	valid_via = {key: via[key] for key in keys[350:400]}

	trainset = ToTensor(RandomTranspose(RandomFilter(ColorJitter(VIADataset(train_via, args.path, shuffle=True)))))
	validset = ToTensor(VIADataset(valid_via, args.path))

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

	model = model.to(device)

	os.makedirs(args.destination, exist_ok=True)
	basename = os.path.join(args.destination, args.name)

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

		print('Training loss = {}'.format(train_losses.mean()))
		print('Validation loss = {}'.format(valid_losses.mean()))

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
