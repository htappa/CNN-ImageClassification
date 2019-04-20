# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os
import glob
import cv2
from torchvision import datasets
from torch.utils.data import DataLoader
# ----------------------------------------------------------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------------------------------------------------------

#This is the directory on my local, but I think it actually needs to be the directory in the cloud.
train_dict = "C:\Users\Kinse\Documents\Google_cloud_Machinelearning\FinalProject-Group4\Code\fruits_data_set\Training"
test_dict = "C:\Users\Kinse\Documents\Google_cloud_Machinelearning\FinalProject-Group4\Code\fruits_data_set\Testing"

#input the folder names of the fruits you want to train/test
targets = ["Apple Red 1", "Cherry", "Grape", "Kiwi", "Quince"]

#Transform the image data to a FloatTensor with shape of (color X height X weight) normalizing along the way
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#Since the dataset is rather large and we do not want to train on all classes
#we simply loop through the target folders and add the images to a train_data and test_data variables
#from torchvision we utilize ImageFolder because it automatically adds the target name to the image data
for target in targets:
    train_data = datasets.ImageFolder('train_dict', transform= transform)
    test_data = datasets.ImageFolder('test_dict', transform= transform)

#Use DataLoader to get batches of data from our datasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle = True)
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
        self.fc = nn.Linear(in_features=100*48*48, out_features=33) #in_features = [(inputsize + 2*pad - kernelsize)/stride] + 1﻿

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

