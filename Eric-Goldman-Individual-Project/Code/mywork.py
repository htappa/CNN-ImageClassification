#-----------------------Group4 Code Contributions-------------------------#
#-------------------------------------------------------------------------#
#-------------------By: Eric Goldman -------------------------------------#
#-------------------------------------------------------------------------#

# Modified the file path
# set file path
train_dict = "./fruits_data_set/Training"
test_dict = "./fruits_data_set/Testing"

#Updated the methodlogy when we decided to use ImageFolder to bring in the pictures
train_data = datasets.ImageFolder(train_dict, transform= transform)
test_data = datasets.ImageFolder(test_dict, transform= transform)

#I performed EDA with the batch size. We decided to use 40 after testing numbers
#bewteen 1 and 100
# define batch size
batch_size = 40

#Performed testing and updated the code based on our changing methodlogies.
# use DataLoader to get batches of data from the datasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = True)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % train_data.classes[labels[j]] for j in range(8)))
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# HYPER-PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------
# I tested multiple hyper-parameter values. We used these three for ease of use and achieved high accuracy
num_epochs = 6
alpha = 0.0001
num_classes = 33

# Worked on saving the matrix to a text file. We ultimately used pandas but test numpy as well.
# print test accuracy
print()
print('Test Accuracy of model on the 5195 test images: %.2f %%' % (100 * correct / total))

# turn confusion matrix into csv
confusion_matrix = pd.DataFrame(confusion_matrix.numpy())
confusion_matrix.to_csv('fruits_cnn_3layer_confmat.csv')
