# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import time

# set seed
torch.manual_seed(10)

# ----------------------------------------------------------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------------------------------------------------------

# set start time to record computation run time
start_time = time.time()

#This is the directory on my local, but I think it actually needs to be the directory in the cloud.
#Updated file path -eg
train_dict = "./fruits_data_subset2/Training"
test_dict = "./fruits_data_subset2/Testing"

#Transform the image data to a FloatTensor with shape of (color X height X weight) normalizing along the way
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder(train_dict, transform= transform)
test_data = datasets.ImageFolder(test_dict, transform= transform)

#Use DataLoader to get batches of data from our datasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=40, shuffle = True)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % train_data.classes[labels[j]] for j in range(40)))
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# HYPER-PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

num_epochs = 6
batch_size = 10
alpha = 0.0001
input_size = 10
hidden_size = 10
num_classes = 5

# ----------------------------------------------------------------------------------------------------------------------
# CONVOLUTIONAL NEURAL NETWORK (CNN) MODEL
# ----------------------------------------------------------------------------------------------------------------------

# 2 layer convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=40000, out_features=num_classes) #in_features = [(inputsize + 2*pad - kernelsize)/stride] + 1﻿

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cnn = CNN()
cnn.cuda()

# ----------------------------------------------------------------------------------------------------------------------
# LOSS & OPTIMIZER
# ----------------------------------------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=alpha)

# ----------------------------------------------------------------------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------------------------------------------------------------------

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # forward pass, backpropagation, optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))

# ----------------------------------------------------------------------------------------------------------------------
# TEST MODEL
# ----------------------------------------------------------------------------------------------------------------------

# change model to 'eval' mode
cnn.eval()
correct = 0
total = 0

# define blank confusion matrix
confusion_matrix = torch.zeros(num_classes, num_classes)

for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    for t, p in zip(labels.view(-1), predicted.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

print()
print('Test Accuracy of model on the 814 test images: %d %%' % (100 * correct / total))

# turn confusion matrix into csv
confusion_matrix = pd.DataFrame(confusion_matrix.numpy())
confusion_matrix.to_csv('testset2_conf.csv')

# print run time
print('fruits_cnn_subset2 run time: %s seconds' % (time.time() - start_time))
