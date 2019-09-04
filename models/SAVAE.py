# -*- coding: utf-8 -*-
"""
Self-attention VAE

References:
[1] Wang, Xiaolong, et. al. "Non-local neural networks." CVPR, 2018.
[2] Han Zhang, et. al., "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018)
[3] A. Vaswani, et. al. "Attention is all you need." arXiv:1706.03762, 2017.
"""

import sys
sys.path.append(r'C:\AI, Machine learning')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os          
import time
from utility import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Output directory
output_dir = 'DRAW_out/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

chkpt_dir = 'DRAW_chkpt/'
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)
    
###############################
# Hyperparameters
###############################

# Dataset
batch_size = 128
    
# Number of units in each hidden layer (excl. latent code layer)
h_len = 1024

# Length of latent code (number of units in latent code layer)
z_len = 128

# Training
num_epochs = 100
lr         = 3e-4

###############################
# Load data
###############################

# MNIST dataset: 60,000 training, 10,000 test images.
# We'll take NUM_VAL of the training examples and place them into a validation dataset.
NUM_TRAIN  = 55000
NUM_VAL    = 5000

# Training set
mnist_train = dset.MNIST('C:/datasets/MNIST',
                         train=True, download=True,
                         transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
# Validation set
mnist_val = dset.MNIST('C:/datasets/MNIST',
                       train=True, download=True,
                       transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
# Test set
mnist_test = dset.MNIST('C:/datasets/MNIST',
                       train=False, download=True,
                       transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# Input dimensions
train_iter     = iter(loader_train)
images, labels = train_iter.next()
_, C, H, W     = images.size()
img_flat_dim   = H * W          # Flattened image dimension

###############################
# Model
###############################

# Self-attention mechanism
# We follow the variable naming convention in [2]
class SelfAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.zeros(1))

        # 1x1 convolution layers. [2] uses out_channels = in_channels / 8
        # for f and g.
        scale = 8
        if (in_channels < scale):
            scale = 1
        self.conv_f = nn.Conv2d(in_channels, in_channels // scale, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels, in_channels // scale, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        mb_size, in_C, in_H, in_W = x.size()
        in_N = in_H * in_W
        
        f = self.conv_f(x)  # Query
        g = self.conv_g(x)  # Key
        h = self.conv_h(x)  # Value
        
        # Reshape to turn Query and Key into 2D matrices (see [1], Fig. 2):
        # Keep channel dimension but flatten height & width into one dimension
        f = f.view(mb_size, -1, in_N)
        g = g.view(mb_size, -1, in_N)
        h = h.view(mb_size, -1, in_N)
        
        # Attention map, b
        # Each row of b sums to 1, so each element represents the relative amount
        # of attention (total attention being 1). I.e. b[i, j] is the amout of
        # attention the model pays to the j-th location when synthesizing the
        # i-th position.
        s = torch.matmul(f.transpose(1, 2), g)  # Tensor size: (H*W, H*W)
        # REVISIT: Try the dot-product alternative, Eq. (4) in [1]
        b = F.softmax(s, dim=1)                 # Each row sums to 1

        o = torch.matmul(h, b.transpose(1, 2)).view(mb_size, in_C, in_H, in_W)
        
        # REVISIT: Try introducing a parameter to scale the residual connection
        # REVISIT: so the model can learn to reduce the residual contribution to
        # REVISIT: 0 if needed (initialise the scale parameter to 1). Perhaps try
        # REVISIT: y = self.gamma * o + (1 - self.gamma) * x
        # Non-local block: non-local op (attention) + residual connection [1]
        y = self.gamma * o + x
        
        return y

class SAVAE(nn.Module):
    def __init__(self, C, H, W, h_len, z_len, use_attn=True):
        super().__init__()

        self.encoder = nn.Sequential(
                           SelfAttn(1),
                           nn.Conv2d(1, 32, kernel_size=5, stride=1),
                           nn.LeakyReLU(inplace=True), # Default negative slope is 0.01
                           nn.MaxPool2d(2, stride=2),
                           nn.Conv2d(32, 64, kernel_size=5, stride=1),
                           nn.LeakyReLU(inplace=True),
                           #nn.MaxPool2d(2, stride=2),
                           SelfAttn(64),
                           Flatten(),
                           nn.Linear(4096, h_len),
                           nn.LeakyReLU(inplace=True)
                       )
        
        self.fc_mean = nn.Linear(h_len, z_len)
        self.fc_var  = nn.Linear(h_len, z_len)
        
        self.decoder = nn.Sequential(
                           nn.Linear(z_len, 1024),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm1d(1024),
                           nn.Linear(1024, 6272),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm1d(6272),
                           Unflatten(-1, 128, 7, 7),
                           nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(64),
                           nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                           nn.Sigmoid(),
                       )
    
    def reparameterize(self, mean, logvar):
        sd  = torch.exp(0.5 * logvar)   # Standard deviation
        # Latent distribution uses Gaussian
        eps = torch.randn_like(sd)
        z   = eps.mul(sd).add(mean)
        return z
    
    def forward(self, x):
        enc    = self.encoder(x)
        mean   = self.fc_mean(enc)
        logvar = self.fc_var(enc)
        z      = self.reparameterize(mean, logvar)
        dec    = self.decoder(z)

        return dec, mean, logvar
    
    def generate_img(self, z):
        dec = self.decoder(z)

        return dec
    
###############################
# Loss function
###############################

# The Evidence Lower Bound (ELBO) gives the negative loss, so
# minimising the loss maximises the ELBO
def loss_fn(x_original, x_recon, mean, logvar):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x_original, reduction='sum')

    # KL divergence has the following closed-form solution if we assume a
    # Gaussian prior and posterior:
    # -( 0.5 * sum(1 + log(sd ** 2) - mean ** 2 - sd ** 2) )
    # Derivation in Appendix B of VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KL_loss = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp())
    
    return recon_loss + KL_loss

###############################
# Main
###############################
        
model     = SAVAE(C, H, W, h_len, z_len).to(device)
optimiser = optim.Adam(model.parameters(), lr=lr)

def train(epoch):
    start_time = time.time()
    model.train()
    loss_train = 0
    for batch, (x, _) in enumerate(loader_train):
        with torch.autograd.detect_anomaly():
            recon, mean, logvar = model(x)
            loss        = loss_fn(x, recon, mean, logvar)
            loss_train += loss.item()
              
            loss.backward()
            
        optimiser.step()
        optimiser.zero_grad()

    epoch_time = time.time() - start_time
    print("Time Taken for Epoch %d: %.2fs" %(epoch, epoch_time))
    
    if epoch % 2 == 0:
        print("Epoch {} reconstruction:".format(epoch))
        imgs_numpy = recon.detach().to('cpu').numpy()
        fig = show_images(imgs_numpy[0:16])
        plt.close(fig)
    
    print('Epoch {} avg. training loss: {:.3f}'.format(epoch, loss_train / len(loader_train.dataset)))

def validation(epoch):
    model.eval()
    loss_val = 0
    with torch.no_grad():
        for batch, (x, _) in enumerate(loader_val):
            recon, mean, logvar = model(x)
            loss      = loss_fn(x, recon, mean, logvar)
            loss_val += loss.item()
            
    print('Epoch {} validation loss: {:.3f}'.format(epoch, loss_val / len(loader_val.dataset)))

for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    train(epoch)
    validation(epoch)
    if epoch % 2 == 0:
        with torch.no_grad():
            print("Epoch {} generation:".format(epoch))
            z      = torch.randn(32, z_len).to(device)
            sample = model.generate_img(z)
            imgs_numpy = sample.to('cpu').numpy()
            fig = show_images(imgs_numpy[0:16])
            # Save image to disk
            fig.savefig('{}/{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        # Save checkpoints
        torch.save({
            'model'      : model.state_dict(),
            'optimiser'  : optimiser.state_dict(),
            'hyperparams': {'h_len'      : h_len,
                            'z_len'      : z_len,
                            'num_epochs' : num_epochs}
        }, '{}/epoch_{}.pth'.format(chkpt_dir, epoch))

