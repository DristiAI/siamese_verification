import keras
from keras.models import Model
from keras.layers import Input,Conv2D, MaxPooling2D, BatchNormalization,Flatten,Dense
import keras.backend as K

def model(input_shape):

    """
    SIAMESE NET IMPLEMENTATION

    """
    inp= Input(shape=input_shape)
    #Convolution layer 1
    x= Conv2D(16,(3,3),activation='relu',padding='same')(inp)
    x= BatchNormalization()(x)
    x= MaxPooling2D()(x)
    #Convolution layer 2
    x= Conv2D(32,(3,3),activation='selu',padding='same')(x)
    x= BatchNormalization()(x)
    x= MaxPooling2D()(x)
    #Convolution layer 3
    x= Conv2D(64,(3,3),activation='selu',padding='same')(x)
    x= BatchNormalization()(x)
    #Convolution layer 4
    x= Conv2D(128,(3,3),activation='selu',padding='same')(x)
    x= MaxPooling2D()(x) 
    x= Flatten()(x)
    #Embeddings we will finally be using 
    x= Dense(128,activation='selu')(x)
    
    model= Model(inputs=inp,outputs=x)
    
    return model


#LOSS FUNCTION

def contrastive_loss(y_true,y_pred,margin=1):
    """
    Taken: siamian mnist example in keras 
    
    """

    return K.mean((1-y_true)*K.square(y_pred)+(y_true)*K.square(K.maximum(margin-y_pred,0)))


def euclidean_distance(y):
    y1,y2=y
    return K.sqrt(K.sum(K.square(y1-y2),axis=1,keepdims=True))



if __name__== '__main__':
    
    from keras.utils import plot_model
    model = model((56,46,1))
    print(model.summary)
    plot_model(model,to_file='./images/model.png')
    
