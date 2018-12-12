import numpy as np
import os
import random
import shutil
import cv2

"""
Enter the DATA_PATH  as the path where your training set is

"""
DATA_PATH = './att_faces'
TRAIN_PATH = './train'
VALIDATION = './val'
TEST_PATH = './test'
NUM_CLASSES = 32
input_shape=(56,46,1)

def create_paths(*paths):
    """
    create paths 
    """

    for i in paths:
        if os.path.exists(i): continue
        os.makedirs(i)


def copy_files(folders,path):
    
    for folder in folders:
        dest=path+'/'+str(folder)
        src= os.path.join(DATA_PATH,folder)
        os.makedirs(dest)
        for i in os.listdir(src):
            shutil.copy(src+'/'+str(i),dest)


def make_dataset(path):
    folders= os.listdir(path)
    y=[]
    x=[]
    for folder in folders:
        
        files=os.listdir(os.path.join(path,folder))
        
        for file in files:
            #print(os.path.join(path,folder,file))
            y+=[folder]
            
            """
            Before adding to the data 
            * read the image
            * scale down by 255
            * resize the image
            * make the image 3d 
            
            """
            x+= [cv2.resize(cv2.imread(os.path.join(path,folder,file),0)/255,(56,46)).reshape(56,46,1)]
    print(len(x))
    return x,y

def create_input(X,indices):
    """
    This method is used only for creating pairs for training and validation
    
    
    returns: x = a list with each element as [img1,img2] 
             y = a list of labels 1s and 0s
             
        
               
    """
    x=[]
    y=[]
    
    print("inside the create_input function now")
    for i in range(NUM_CLASSES):
        for j in range(9):
            i1,j1 = indices[i][j],indices[i][j+1]
            #print(i1,j1)
            #print(X.shape)
            img1= X[i1]
            
            img2= X[j1]
            #print(img1)
            x+=[[img1,img2]]
            y+=[0]
            inc=random.randrange(1,NUM_CLASSES)
            dn =(i+inc)%NUM_CLASSES
            i1,j2=indices[i][j],indices[dn][j]
            x+=[[X[i1],X[j2]]]
            y+=[1]
            
    XY = list(zip(x,y))
    random.shuffle(XY)
    x,y = zip(*XY)
    
    return x,y
