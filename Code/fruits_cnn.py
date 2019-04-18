# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# HYPER-PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

num_epochs = 10
batch_size = 100
alpha = 0.0001

# ----------------------------------------------------------------------------------------------------------------------
# CONVOLUTIONAL NEURAL NETWORK (CNN) MODEL
# ----------------------------------------------------------------------------------------------------------------------

# 2 layer network
# batch shape for input x is (3, 100, 100)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=50, kernel_size=10, padding=2),
            nn.BatchNorm2d(num_features=50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=10, padding=2),
            nn.BatchNorm2d(num_features=100),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=500, out_features=33)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ----------------------------------------------------------------------------------------------------------------------
# LOSS & OPTIMIZER
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# TEST MODEL
# ----------------------------------------------------------------------------------------------------------------------

