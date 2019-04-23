#-----------------------Group4 Code Contributions-------------------------#
#-------------------------------------------------------------------------#
#-------------------By: Ryan Kinsey (pkinsey@gwu.edu)---------------------#
#-------------------------------------------------------------------------#

#Please note that the all of my same code below was contributed to all 3 of our .py files, since the files are quite similar.
#25 lines of code total (per .py file)
#2 from the internet (Denoted below with #INTERNET CODE# at the end)
#The rest was written independently or modified and fitted from class work



#I was tasked with coding the LOAD DATASET section
#The 12 lines below:
# 1) fetch the data from our GCP folder
# 2) normalize and transform the images coming in and then convert to a Tensor
# 3) create a pytorch image folder (took a bit of research to get the hang ImageFolder.())
# 4) create a train/test dataloder (again, spent time researching the source code to understand DataLoader.())
# 5) print the number of images in the training & testing set for convenience

# ----------------------------------------------------------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------------------------------------------------------

# set file path
train_dict = "./fruits_data_set/Training"
test_dict = "./fruits_data_set/Testing"

# transform image data to FloatTensor with shape of (color x height x weight) normalizing along the way
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #INTERNET CODE#

train_data = datasets.ImageFolder(train_dict, transform= transform)
test_data = datasets.ImageFolder(test_dict, transform= transform)

# define batch size
batch_size = 40

# use DataLoader to get batches of data from the datasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = True)
print('Size of Training Set: ' , (len(train_loader.dataset)))
print('Size of Testing Set: ' , (len(test_loader.dataset)))


# ----------------------------------------------------------------------------------------------------------------------
# MISCELLANEOUS CODE
# ----------------------------------------------------------------------------------------------------------------------

#Although I was not tasked with any other coding sections,
#I found myself adding small, little, things throughout the code for convenience.
#Additionally, I contributed to the loss & accuracy visualizations.

#Printing the model architecture upon model initialization.
print('CNN Architecture: ' , (cnn))

#loss.data() had a deprecation warning, changed to loss.item() which helped with visualizations too.
loss_list.append(loss.item()) #INTERNET CODE#

#Printing out the accuracy and run time to 2 decimal places.
print('Test Accuracy of model on the 5195 test images: %.2f %%' % (100 * correct / total))
print('fruits_cnn_3layer run time: %.2f seconds' % (time.time() - start_time))

#Helped format the plots, specifically the x-axis.
#Based on how our for-loop was structured,
#I figured out what data was being used for the x-axis
#and appropriately formatted the tick marks and labels.
plt.xlabel("Epochs")
#The x-axis represents the number of epochs
#It is broken down as ((number of images) / (minibatch size)) * epochs
#As such, the below line creates the appropriate xticks
plt.xticks(np.arange(0, 2325, step=387), ('0','1', '2', '3', '4', '5','6'))
plt.ylabel("Loss")
#Change y scale to log to make it look better
plt.yscale('log')
plt.show()

#Similar idea as above, this time for the accuracy graph.
plt.xlabel("Epochs")
#The x-axis represents the number of epochs.
#It is broken down as the ((number of images) / (minibatch size)) * epochs
#As such, the below line creates the appropriate xticks.
plt.xticks(np.arange(0, 130, step=22), ('0','1', '2', '3', '4', '5','6'))
plt.ylabel("Accuracy (%)")
plt.show()
