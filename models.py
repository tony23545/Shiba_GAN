import torch
import torch.nn as nn

# number of channels in the training image
nc = 3

# size of latent vector
nz = 100

# size of feature maps in generator
ngf = 64

# size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input size is Z, going into a convolution
			nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False), 
			nn.BatchNorm2d(ngf * 8), 
			nn.ReLU(True), 
			# size: ngf*8 * 4 * 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False), 
			nn.BatchNorm2d(ngf * 4), 
			nn.ReLU(True),
			# size: ngf*4 * 8 * 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ngf * 2), 
			nn.ReLU(True),
			# size: ngf*2 * 16 * 16
			nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ngf), 
			nn.ReLU(True),
			# size: ngf * 32 * 32
			nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# size: ngf * 64 * 64
			nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
			nn.Tanh()
			# size: nc * 128 * 128
		)

	def forward(self, input):
		return self.main(input)


class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is nc * 128 * 128
			nn.Conv2d(nc, ndf, 4, 2, 1, bias = False), 
			nn.LeakyReLU(0.2, inplace = True), 
			# size: ndf * 64 * 64, 
			nn.Conv2d(ndf, 2 * ndf, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf * 2), 
			nn.LeakyReLU(0.2, inplace = True), 
			# size: 2*ndf * 32 * 32
			nn.Conv2d(2 * ndf, 4 * ndf, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace = True),
			# size: ndf*4 * 16 * 16
			nn.Conv2d(4 * ndf, 8 * ndf, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf * 8), 
			nn.LeakyReLU(0.2, inplace = True),
			# size: ndf*8 * 8 * 8
			nn.Conv2d(8 * ndf, 8 * ndf, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf * 8), 
			nn.LeakyReLU(0.2, inplace = True), 
			# size: ndf*8 * 4 * 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False)
			nn.Sigmoid()
		)

	def forward(self, input):
		return self.main(input)