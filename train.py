from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from models import *


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# configuration
# root dir for dataset
dataroot = "data/shiba"

# number of workers for dataloader
workers = 2

# batch size during training
batch_size = 16

# number of training epochs
num_epochs = 5

# learning rate
lr = 0.0002

# beta1 for adam optimizers
beta1 = 0.5

# number of GPU available 
ngpu = 1

nz = 100

# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

def main():
	dataset = dset.ImageFolder(root = dataroot, 
							   transform = transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
								]))
	# create dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = workers)

	# decide which device to use
	device = torch.device("cuda 0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

	# Plot some training images
	real_batch = next(iter(dataloader))
	plt.figure(figsize=(8,8))
	plt.axis("off")
	plt.title("Training Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

	# create the generator
	netG = Generator(ngpu).to(device)
	netG.apply(weight_init)
	print(netG)

	# create the discriminator
	netD = Discriminator(ngpu).to(device)
	netD.apply(weights_init)
	print(netD)

	# loss function
	criterion = nn.BCELoss()

	fixed_noise = torch.randn(64, nz, 1, 1, device=device)
	# establish conventions for real and fake labels during training
	real_label = 1
	fake_label = 0

	# setup Adam optimizers for both G and D
	optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))

	# training
	img_list = []
	G_losses = []
	D_losses = []
	iters = 0
	print("start training...")

	for epoch in range(num_epochs):
		for i, data in enumerate(dataloader, 0):
			#######################
			# update D network: maximize log(D(x)) + log(1 - D(G(z)))
			#######################
			netD.zero_grad()
			real_cpu = data[0].to(device)
			b_size = real_cpu.size(0)
			label = torch.full((b_size, ), real_label, device = device)
			# forward pass real batch
			output = netD(real_cpu).view(-1)
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()

			# train with fake data
			noise = torch.randn(b_size, nz, 1, 1, device = device)
			fake = netG(noise)
			label.fill_(fake_label)
			output = netD(fake.detach()).view(-1)
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			# cumulate error
			errD = errD_real + errD_fake
			optimizerD.step()

			########################
			# update G network: maximize(log(D(G(z)))
			########################
			netG.zero_grad()
			label.fill_(real_label)
			output = netD(fake).view(-1)
			errG = criterion(output, label)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			# output training stats
			if i % 10 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
 					% (epoch, num_epochs, i, len(dataloader),
						errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

			# save losses
			G_losses.append(errG.item())
			D_losses.append(errD.item())

			# Check how the generator is doing by saving G's output on fixed_noise
			if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
				with torch.no_grad():
					fake = netG(fixed_noise).detach().cpu()
				img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

			iters += 1
	
	# plot result
	plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="D")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()

	# save model
	torch.save(netG.state_dict(), "generator.pt")
	torch.save(netD.state_dict(), "discriminator.pt")
	print("finish training")

if __name__ == "__main__":
	main()

