"""
"""

###########
# Imports #
###########

import copy
import numpy as np
import time
import torch


#############
# Functions #
#############

def train_epoch(model, loader, criterion, optimizer):
	model.train()
	losses = list()

	for inputs, targets in loader:
		inputs = inputs.cuda()
		targets = targets.cuda()
		outputs = model(inputs)

		loss = criterion(outputs, targets)
		losses.append(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return np.mean(losses), np.std(losses)

def eval(model, loader, criterion):
	model.eval()
	losses = list()

	with torch.no_grad():
		for inputs, targets in loader:
			inputs = inputs.cuda()
			targets = targets.cuda()
			outputs = model(inputs)

			loss = criterion(outputs, targets)
			losses.append(loss.item())

	return np.mean(losses), np.std(losses)

def train(model, loaders, criterion, optimizer, epochs=10):
	best_model = None
	best_loss = 1e10

	for epoch in range(epochs):
		print('Epoch {}'.format(epoch + 1))
		print('-' * 10)

		start = time.time()

		# Training
		mean, std = train_epoch(model, loaders[0], criterion, optimizer)
		print('Training loss = {} +- {}'.format(mean, std))

		# Validation
		mean, std = eval(model, loaders[1], criterion)
		print('Validation loss = {} +- {}'.format(mean, std))

		if mean < best_loss:
			best_loss = mean
			best_model = copy.deepcopy(model.state_dict())
			print('New best model')

		elapsed = time.time() - start

		print('{:.0f}m{:.0f}s elapsed'.format(elapsed // 60, elapsed % 60))
		print('-' * 10)

	return best_model


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import json
	import os

	from torch.utils.data import DataLoader
	from torch.optim import Adam
	from datetime import datetime

	from dataset import LargeDataset
	from models import UNet
	from criterions import DiceLoss

	# Datasets
	with open('products/json/data.json', 'r') as f:
		data = json.load(f)

	np.random.seed(0)
	np.random.shuffle(data)

	for _, _, boxes in data:
		np.random.shuffle(boxes)

	trainset = LargeDataset(data[:350], origin='resources/')
	validset = LargeDataset(data[350:400], origin='resources/')
	testset  = LargeDataset(data[400:450], origin='resources/')

	# Dataloaders
	trainloader = DataLoader(trainset, batch_size=5)
	validloader = DataLoader(validset, batch_size=5)
	testloader = DataLoader(testset, batch_size=1)

	# Model
	model = UNet(3, 1)
	model.cuda()

	# Criterion and optimizer
	criterion = DiceLoss()
	optimizer = Adam(model.parameters(), lr=1e-3)

	# Training
	best = train(model, (trainloader, validloader), criterion, optimizer, 10)
	model.load_state_dict(best)

	date = datetime.now().strftime('%Y-%m-%d_%Hh%M')
	os.makedirs('products/models/', exist_ok=True)
	torch.save(best, 'products/models/unet_' + date + '.pth')

	# Testing
	mean, std = eval(model, testloader, criterion)
	print('Testing loss = {} +- {}'.format(mean, std))
