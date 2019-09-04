# -*- coding: utf-8 -*-
"""
Pytorch implementation of Spatial Transformer Network

References:
[1] https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
[2] Jaderberg et al. "Spatial Transformer Networks." NIPS 2015. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import os          
import time
from utility import *


plt.ion()   # interactive mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
output_dir = 'STN_out/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
chkpt_dir = 'STN_chkpt/'
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)
    
###############################
# Hyperparameters
###############################

# Dataset
batch_size = 64

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

class STN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN layers for MNIST classification
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network (just a regular CNN)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # "Regressor" for the 3 * 2 affine transformation matrix (the transformation parameters)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        # Transformation parameters
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # affine_grid() produces a 'flow field' in the output space. Given an affine
        # transformation matrix and a regular grid in the output space, the function
        # works out the warped sampling grid in the input space. Since the warped
        # grid doesn't necessarily have a 1-to-1 correspondence with the input pixels,
        # interpolation is required and the 'flow field' specifies the interpolation
        # for each pixel in the output.
        # grid_sample() performs the interpolation accordingly.
        grid = F.affine_grid(theta, x.size())
        xt   = F.grid_sample(x, grid)

        return xt

    def forward(self, x):
        # Spatial transform the input
        x = self.stn(x)

        # Classification network forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # Output the log likelihood of each label class
        ll = F.log_softmax(x, dim=1)
        
        return ll


###############################
# Main
###############################

model     = STN().to(device)
optimizer = optim.SGD(model.parameters(), lr=3e-4)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader_train):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss   = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_train.dataset),
                100. * batch_idx / len(loader_train), loss.item()))

def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct   = 0
        for data, target in loader_test:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Get the index of the max log-probability
            pred       = output.max(1, keepdim=True)[1]
            correct   += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader_test.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(loader_test.dataset),
                      100. * correct / len(loader_test.dataset)))

for epoch in range(1, 20 + 1):
    train(epoch)
    test()
    
# Visualizing STN Results

# Helper function
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp  = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp  = std * inp + mean
    inp  = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.
def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(loader_test))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
                        torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()