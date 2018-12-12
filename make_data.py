"""
Run this script to create the data folders 
./Train
./val
./test

This script requires that you have a dataset with all the images of same classes
in  sub-folders with name of the class as the sub-folders' name.


"""

from data import *

#total classes in given dataset
folders= [i for i in files if i.startswith('s')]
num_classes_original = len(folders)
training_sampling_ratio = 0.8 
num_train_folders = int(0.8 *num_classes_original) 

train_folders = np.random.choice(folders,num_train_folders,replace=False).tolist()
remaining_folders= np.setdiff1d(folders,train_folders)
validation_folders = [remaining_folders[i] for i in range(4)]
test_folders= np.setdiff1d(remaining_folders,validation_folders)

copy_files(train_folders,TRAIN_PATH)
copy_files(validation_folders,VALIDATION) #don't change this order
copy_files(test_folders,TEST_PATH)


if not os.path.exists('./images'):
    os.makedirs('./images')






