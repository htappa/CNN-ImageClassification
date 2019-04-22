# Fruits Classification Using CNN

### Authors
Henry Tappa, M.S.  
Ryan Kinsey, M.S.  
Eric Goldman, M.S.  

### Description
The purpose of this project is to build a deep learning nerual network that can accurately classify images of different fruits. Potential applications of this network include automating Customs and Border Protections processes or for use in the agricultural industry.

### The Dataset
The dataset being used is a collection of high-quality images containing different fruits. The dataset was obtained from Kaggle.com and created by Horea Muresan and Mihai Oltean (citation below). The pictures were taken on a white background and the camera was rotated incrementally 360° around the fruit taking pictures every 20°.
- Training set: 15506 images
- Testing set: 5195 images
- Number of classes: 33 fruits
- Image size: 100x100 pixels
- Filename format: image_index_100.jpg. "100" represents image size (100x100 pixels).

Different varieties of the same fruit are shown having different labels (e.g. Apple Red 1, Apple Red 2).

### The Model
The neural network model being implemented for this classificaiton task is a Convolutional Neural Network (CNN) built using Pytorch. We first tested a 2-layer CNN on a subset of the data, then used the same network on the the full dataset. A 3-layer CNN was then built and tested on the full dataset. Each of these networks can be found in three seperate python files:
- fruits_cnn_subset.py: 2-layer CNN used on a subset of the data. 5 classes. 100% accuracy.
- fruits_cnn_all_2layer.py: 2-layer CNN used on the full dataset. 33 classes. 97% accuracy.
- fruits_cnn_all_3layer.py: 3-layer CNN used on the full dataset. 33 classes. 99% accuracy.

Each of these networks are set up to run CUDA operations, and our test machine utilized an NVIDIA Tesla K80 GPU. 

Further description of the model and results can be found in the project report, "Group4ProjectReport.pdf."

### Dependencies
CUDA compatible GPU  
Python 3.5.2  
Pytorch 0.4.1  
Pandas 0.23.4  
Numpy 1.15.0  
Matplotlib 1.5.1  

### Citations
Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Technical Report, Babes-Bolyai University, 2017
