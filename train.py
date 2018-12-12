from model import *
from data import *
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
import cv2


IMAGE_IN_A_CLASS =10
INPUT_SHAPE= (56,46,1) #image size

train_x,train_y = make_dataset(TRAIN_PATH)
val_x,val_y =  make_dataset(VALIDATION)
test_x,test_y = make_dataset(TEST_PATH)

def preprocess_dataset(*data_matrices):
    
    """
    This function is only for the dataset during training process .
    It assumes the step of resizing the images to 56 px X 46 px has been
    done already
    
    """
    output=[]
    for data_matrix in data_matrices:
        data_matrix =np.array(data_matrix)/255
        output.append(data_matrix)
    return output

train,val,test = map(np.array ,preprocess_dataset(train_x,val_x,test_x))
train_y,val_y,test_y = map(np.array,[train_y,val_y,test_y])

T_CLASSES= set(train_y)
V_CLASSES = set(val_y)


train_indices = [np.where(train_y==i)[0] for i in T_CLASSES]
val_indices = [np.where(val_y==i)[0] for i in V_CLASSES]

"""

CREATING THE DATA TO FEED TO THE MODEL
T_INPUT[i] = [image1,image2]
T_LABELS = [0] WHEN image1 and image2 are same person/
           [1] WHEN image1 and image2 are different person
"""

T_INPUT,T_LABELS = map(np.array,create_input(train,train_indices))
V_INPUT,V_LABELS = map(np.array,create_input(train,train_indices))

INPUT1 = Input(shape=INPUT_SHAPE)
INPUT2 = Input(shape=INPUT_SHAPE)

model_ = model(input_shape)

SIAMESE_NET1_out = model_(INPUT1)
SIAMESE_NET2_out = model_(INPUT2)

#output is the euclidean distance of the two inputs

output = Lambda(euclidean_distance)([SIAMESE_NET1_out,SIAMESE_NET2_out])
label = Dense(1,activation='sigmoid')(output)
my_Callback = ModelCheckpoint(filepath='model_fac.hdf5', verbose=1, save_best_only=True)
MODEL = Model(inputs= [INPUT1,INPUT2], outputs= label)
MODEL.compile(loss=contrastive_loss, optimizer='adamax',metrics=['accuracy'])
MODEL.fit([T_INPUT[:,0],T_INPUT[:,1]],T_LABELS,batch_size=128,epochs=80,validation_data=([V_INPUT[:,0],V_INPUT[:,1]],V_LABELS),callbacks=[my_Callback])
