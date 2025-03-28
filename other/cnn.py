import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

import skimage
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_mnist = datasets.MNIST(
#     root='dataset/',
#     train=True,
#     transform=Compose([RandomCrop(24), ToTensor()]),
#     download=True)
# test_mnist = datasets.MNIST(
#     root='dataset/',
#     train=False,
#     transform=Compose([RandomCrop(24), ToTensor()]),
#     download=True)

# train_dataset = datasets.KMNIST(
#     root='dataset/',
#     train=True,
#     transform=ToTensor(),
#     download=True)
# test_dataset = datasets.KMNIST(
#     root='dataset/',
#     train=False,
#     transform=ToTensor(),
#     download=True)

# rand_int = np.random.choice(len(train_dataset))
# plt.imshow(train_dataset[rand_int][0][0], cmap='Greys_r')


# cameraman = Image.fromarray(skimage.data.camera())
# transform = Compose([
#     ToTensor(),
#     Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
# ])
# cameraman = transform(cameraman)

# plt.imshow(cameraman[0], cmap='Greys_r')
 
# horiz_filter = torch.tensor([[[
#     [1.0, 2.0, 1.0],
#     [0.0, 0.0, 0.0],
#     [-1.0, -2.0, -1.0]
# ]]])

# filtered = F.conv2d(cameraman, horiz_filter)
# plt.imshow(filtered[0],  cmap='Greys_r')



# vert_filter = torch.tensor([[[
#     [1.0, 0.0, -1.0],
#     [2.0, 0.0, -2.0],
#     [1.0, 0.0, -1.0]
# ]]])

# filtered = F.conv2d(cameraman, vert_filter)
# plt.imshow(filtered[0],  cmap='Greys_r') 

# average_filter = torch.ones((1, 1, 7, 7)) / 49.

# filtered = F.conv2d(cameraman, average_filter)
# plt.imshow(filtered[0],  cmap='Greys_r')
# plt.show()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).init()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=2)
        self.linear = nn.Linear(20,10)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.squeeze()
        x = self.linear(x)

        return x
