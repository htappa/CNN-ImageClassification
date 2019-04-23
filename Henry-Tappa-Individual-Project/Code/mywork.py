# NOTE: This is simply a list of code that I wrote for the project, and will not run on it's own in this file.
# ----------------------------------------------------------------------------------------------------------------------

# Packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# set seed
torch.manual_seed(9)


# ----------------------------------------------------------------------------------------------------------------------
# Load Dataset

# set start time to record computation run time
start_time = time.time()


# ----------------------------------------------------------------------------------------------------------------------
# CNN Model

# 2 layer convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # 3 input channels, given that we are using color images (R, G, B), and 16 output channels
            # using 5x5 kernel with padding=1 to ensure that kernel properly passes over edges of image
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            # apply batch normalization
            nn.BatchNorm2d(num_features=32),
            # apply relu
            nn.ReLU(),
            # apply 2D max pooling over feature set
            nn.MaxPool2d(kernel_size=2, stride=2))
        # repeat with second layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # repeat with third layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # pass to fully connected layer
        self.fc = nn.Linear(in_features=18432, out_features=num_classes)

    def forward(self, x):
        # compute first layer
        out = self.layer1(x)
        # compute second layer
        out = self.layer2(out)
        # compute third layer
        out = self.layer3(out)
        # reshape data
        out = out.view(out.size(0), -1)
        # compute fully connected layer
        out = self.fc(out)
        return out

# define cnn and set to run on GPU
cnn = CNN()
cnn.cuda()


# ----------------------------------------------------------------------------------------------------------------------
# Loss & Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=alpha)


# ----------------------------------------------------------------------------------------------------------------------
# Train Model

# define blank list for plotting loss
loss_list = []

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

        #append loss to loss_list
        loss_list.append(loss.item())

        # print loss for iterations in each epoch
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))


# ----------------------------------------------------------------------------------------------------------------------
# Test Model

# change model to 'eval' mode
cnn.eval()

# set correct and total equal to zero
correct = 0
total = 0

# define blank confusion matrix
confusion_matrix = torch.zeros(num_classes, num_classes)

# define blank list for plotting accuracy
accuracy_list = []

for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    # calculate accuracy and append to accuracy_list
    accuracy = 100 * correct / total
    accuracy_list.append(accuracy.data)

    # add label values and predicted values to confusion matrix
    for t, p in zip(labels.view(-1), predicted.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

# print run time
print()
print('fruits_cnn_3layer run time: %.2f seconds' % (time.time() - start_time))
