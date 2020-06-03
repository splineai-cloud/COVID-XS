'''
## © Copyright (C) 2020-2024 Xilinx, Inc
##
## Licensed under the Apache License, Version 2.0 (the "License"). You may
## not use this file except in compliance with the License. A copy of the
## License is located at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
## Developed by SplineAI (www.spline.ai) in Collaboration with Xilinx
'''

# USAGE
# python code/train_COVID.py --network Pnem3 --weights keras_model/COVID/Pnem3 --epochs  50 --init_lr 0.0001 --batch_size 32
# python code/train_COVID.py --network Pnem4 --weights keras_model/COVID/Pnem4 --epochs  50 --init_lr 0.0001 --batch_size 64


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import backend as K
#from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2

from datetime import datetime #DB
from keras.utils import plot_model #DB
from keras.callbacks import ModelCheckpoint #DB
from keras.callbacks import LearningRateScheduler
import os # DB

from random import seed
from random import random
from random import shuffle #DB

import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from keras.models import Model,load_model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, ZeroPadding2D
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, LeakyReLU, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing import image  
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight 
from keras import regularizers
from keras.regularizers import l1
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.resnet50 import  ResNet50


from keras import backend as K

# Setting seeds for reproducibility
seed = 232
##################################################################################

import argparse #DB
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w",  "--weights", default="./keras_model/COVID/Pnem0", help="path to best model HDF5 weights file")
ap.add_argument("-n",  "--network", default="Pnem1",             help="input CNN")
ap.add_argument("-d",  "--dropout", type=int, default=-1,    help="whether or not Dropout should be used")
ap.add_argument("-p",  "--height",  type=int, default=150,    help="image [height, width]")
ap.add_argument("-e",  "--epochs", type=int, default=20,     help="# of epochs") 
ap.add_argument("-bs", "--batch_size", type=int, default=32, help="size of mini-batches passed to network")
ap.add_argument("-l",  "--init_lr", type=float, default=0.0001,  help="initial Learning Rate")
args = vars(ap.parse_args())

weights = args["weights"]
network = args["network"]

##################################################################################
# initialize the number of epochs to train for, base learning rate,
# and batch size
IMAGE_HEIGHT = args["height"]   #150
NUM_EPOCHS = args["epochs"]     #25
INIT_LR    = args["init_lr"]    #0.0001
BATCH_SIZE = args["batch_size"] #32 

#######################################################################################################################################
input_path = './dataset/Pneumonia/covid_data/'
 
mpath3 = './keras_model/Pneumonia/Pnem3/'
fmod3 = os.path.sep.join([mpath3, "best_chkpt.hdf5"]) 
mpath4 = './keras_model/Pneumonia/Pnem4/'
fmod4 = os.path.sep.join([mpath4, "best_chkpt.hdf5"]) 


dlists = ['train', 'val', 'test']
dlists = ['train', 'test']
for _set in dlists:
    n_norm = len(os.listdir(input_path + _set + '/NORMAL'))
    n_infe = len(os.listdir(input_path + _set + '/PNEUMONIA'))
    n_cov = len(os.listdir(input_path + _set + '/COVID')) 
    print('Set: {}, normal images: {}, pneumonia images: {} covid images: {}'.format(_set, n_norm, n_infe, n_cov))
     
def process_data(img_dims, batch_size):
    #Data generation objects
    
    x_train_n, y_train_n = list(), list()
    x_test_n, y_test_n= list(), list()
    #x_val_n, y_val_n = list(), list()
    x_train_p, y_train_p = list(), list()
    x_test_p, y_test_p = list(), list()
    #x_val_p, y_val_p = list(), list()
    x_train_c, y_train_c = list(), list()
    x_test_c, y_test_c = list(), list()
    #x_val_c, y_val_c = list(), list()

    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    
    ipath = input_path + 'train' + '/NORMAL/'
    #print(ipath)
    flist = os.listdir(ipath)  
    for img in flist:
        impath = ipath+img
        try:
            image = cv2.imread(impath) 
            image = cv2.resize(image, (img_dims, img_dims))  
            label = 0
            x_train_n.append(image)
            y_train_n.append(label)
        except:
            print('e') 
    
    ipath = input_path + 'train' + '/PNEUMONIA/'
    #print(ipath)
    flist = os.listdir(ipath) 
    for img in flist:
        impath = ipath +img
        try:
            image = cv2.imread(impath) 
            image = cv2.resize(image, (img_dims, img_dims)) 
            label = 1
            x_train_p.append(image)
            y_train_p.append(label)
        except:
            print('e') 
       
    ipath = input_path + 'train' + '/COVID/'
    #print(ipath)
    flist = os.listdir(ipath) 
    for img in flist:
        impath = ipath +img
        try:
            image = cv2.imread(impath) 
            image = cv2.resize(image, (img_dims, img_dims)) 
            label = 2
            x_train_c.append(image)
            y_train_c.append(label)
        except:
            print('e') 
             
    x_train = x_train_n + x_train_p + x_train_c
    y_train = y_train_n + y_train_p + y_train_c
    
    train_np = list(zip(x_train, y_train))
    shuffle(train_np)
    x_train, y_train = zip(*train_np)
    print("INFO: train #(NORMAL, PNEUMONIA, COVID) ", len(x_train_n), len(x_train_p), len(x_train_c))
    
    ipath = input_path + 'test' + '/NORMAL/'
    #print(ipath)
    flist = os.listdir(ipath)  
    for img in flist:
        impath = ipath+img
        try:
            image = cv2.imread(impath) 
            image = cv2.resize(image, (img_dims, img_dims)) 
            label = 0 
            x_test_n.append(image)
            y_test_n.append(label)
        except:
            print('e') 
    
    ipath = input_path + 'test' + '/PNEUMONIA/'
    #print(ipath)
    flist = os.listdir(ipath)
    for img in flist:
        impath = ipath +img
        try:
            image = cv2.imread(impath) 
            image = cv2.resize(image, (img_dims, img_dims)) 
            label = 1
            x_test_p.append(image)
            y_test_p.append(label)
        except:
            print('e')    
    
    ipath = input_path + 'test' + '/COVID/'
    #print(ipath)
    flist = os.listdir(ipath)
    for img in flist:
        impath = ipath +img
        try:
            image = cv2.imread(impath) 
            image = cv2.resize(image, (img_dims, img_dims)) 
            label = 2
            x_test_c.append(image)
            y_test_c.append(label)
        except:
            print('e')    
            
    x_test = x_test_n + x_test_p + x_test_c
    y_test = y_test_n + y_test_p + y_test_c 
    
    print("INFO: test #(NORMAL, PNEUMONIA, COVID) ", len(x_test_n), len(x_test_p), len(x_test_c))
                  
    len_xtrain = len(x_train)
    len_xtest = len(x_test)  
    
    x_test  = np.asarray(x_test)/255.0
    x_train = np.asarray(x_train)/255.0
    x_val = x_test
    len_xval = len_xtest 
    
    category = 3 
    y_train = to_categorical(y_train, category) 
    y_test = to_categorical(y_test, category) 
    y_val = y_test

    # data genarators
    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,  
        horizontal_flip=True, 
        height_shift_range=0.11,
        width_shift_range=0.11,
        rotation_range=8,
        shear_range=0.2,
        zoom_range=0.3, 
        #brightness_range=(0.9, 1.1), 
        fill_mode='constant')
    
    test_datagen=ImageDataGenerator()

    train_gen = train_datagen.flow(
        x_train, y_train,
        batch_size=BATCH_SIZE)

    test_gen = test_datagen.flow(
        x_test, y_test,
        batch_size=16)
    
    val_gen = test_gen 
        
    return train_gen, val_gen, x_test, y_test, len_xtrain, len_xval

##################################################################################################
# construct the callback to save only the *best* model to disk
# based on the validation loss
fname = os.path.sep.join([weights, "best_chkpt.hdf5"])

#Spline
#p = 8 #The frction p/10 data will be used for training
# Hyperparameters
img_dims = IMAGE_HEIGHT #150
#EPOCHS = 500
batch_size = BATCH_SIZE
print(img_dims) 
print(batch_size)
# Getting the data
train_gen, val_gen, x_test, y_test, len_xtrain, len_xval = process_data(img_dims, batch_size)


##################################################################################################################################################

def model_Pnem3(img_dims):
    inputs = Input(shape=(img_dims, img_dims, 3))

    # Ist conv block
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(inputs) 
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    # 2nd conv block
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # 3rd conv block
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Fourth conv block
    x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # fifth conv block
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # sixth conv block
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # FC layer
    x = Flatten()(x)
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.4)(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.2)(x) 
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.2)(x) 
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    
    # Output layer   
    output = Dense(units=3, activation='softmax', name='dense_out')(x)
    # Creating model and compiling
    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model

####################################################################################################################################################

def model_Pnem4(img_dims):
    
    inputs = Input(shape=(img_dims, img_dims, 3))

    # Ist conv block
    x = Conv2D(filters=64, kernel_size=(4, 4), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(inputs)  
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(4, 4), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    # 2nd conv block
    x = Conv2D(filters=128, kernel_size=(4, 4), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(4, 4), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # 3rd conv block
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Fourth conv block
    x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # fifth conv block
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # sixth conv block
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # FC layer
    x = Flatten()(x)
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.4)(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.2)(x) 
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.2)(x) 
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    
    # Output layer 
    output = Dense(units=3, activation='softmax', name='dense_out')(x)

    # Creating model and compiling
    model = Model(inputs=inputs, outputs=output) 
    model.summary()
    return model

####################################################################################################################################################

# construct the callback to save only the *best* model to disk
# based on the validation loss
fname = os.path.sep.join([weights, "best_chkpt.hdf5"])
print("INFO: Path to save best checkpoints: ", fname) ;
 
####################################################################################################################
# initialize the optimizer and model
print("[INFO] compiling model...")

if network == "Pnem3":
    model = model_Pnem3(img_dims)
    checkpoint = ModelCheckpoint(filepath=fname, monitor='val_acc', save_best_only=True, save_weights_only=False, verbose=1)
    lr_reduce = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, verbose=1, mode='max',min_lr=INIT_LR)
    early_stop = EarlyStopping(monitor='loss', min_delta=0.1, patience=1, mode='min')
    callbacks_list = [checkpoint, lr_reduce]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    #loaded_model = load_model(fmod1)
    loaded_model = load_model(fmod3)
    loaded_model.layers.pop()
    loaded_model.layers.pop() 
    model.set_weights(loaded_model.get_weights())
    cweights = {0:.7, 1:1.4, 2:17.1}

elif network == "Pnem4":
    model = model_Pnem4(img_dims)
    checkpoint = ModelCheckpoint(filepath=fname, monitor='val_acc', save_best_only=True, save_weights_only=False, verbose=1)
    lr_reduce = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, verbose=1, mode='max',min_lr=INIT_LR)
    early_stop = EarlyStopping(monitor='loss', min_delta=0.1, patience=1, mode='min')
    callbacks_list = [checkpoint, lr_reduce]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #loaded_model = load_model(fmod2)
    loaded_model = load_model(fmod4)
    loaded_model.layers.pop()
    loaded_model.layers.pop() 
    model.set_weights(loaded_model.get_weights()) 
    cweights = {0:.7, 1:1.4, 2:17.1}

else:
    printf("ERROR: Wrong model option")
    

# train the network
print("[INFO] training model...")
startTime1 = datetime.now() #DB

# Fitting the model
H = model.fit_generator(
           train_gen, steps_per_epoch=len_xtrain// batch_size, 
           epochs= NUM_EPOCHS, validation_data=val_gen, 
           validation_steps=len_xval// batch_size, verbose=2, callbacks=callbacks_list, class_weight = cweights)
          
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

# save CNN complete model on HDF5 file #DB
fname1 = os.path.sep.join([weights, "final.hdf5"])
model.save(fname1)

fnamem = os.path.sep.join([weights, "best_chkpt.hdf5"])
model.load_weights(fnamem)

##################################################################################################
print("[INFO] evaluating network on Test and Validation datasets...")
# Evaluate model accuracy with test set
VBATCH_SIZE=16
 
classes=['NORMAL', 'PNEUMONIA', 'COVID']
#print('validation Accuracy: %.3f' % scores[1]) #MH
scores = model.evaluate(x_test, y_test, batch_size=VBATCH_SIZE, verbose=2) #MH 

print('Test Loss: %.3f'     % scores[0]) #MH
print('Test Accuracy: %.3f' % scores[1]) #MH

# make predictions on the test set
preds = model.predict(x_test)

# show a nicely formatted classification report

from sklearn.metrics import accuracy_score, confusion_matrix
# make predictions on the test set
preds = model.predict(x_test)

print(classification_report(y_test, preds.round(), target_names=['NORMAL', 'PNEUMONIA', 'COVID'])) 
#print(classification_report(y_test, preds.round()))

output1 = []
out_test = []
for i in range(len(preds)):
    output1.append(list(preds[i]).index(max(preds[i])))
for i in range(len(y_test)):
    out_test.append(list(y_test[i]).index(max(y_test[i])))

print(list(np.round(output1)).count(0)/len(output1))
print(list(out_test).count(0)/len(out_test))

acc = accuracy_score(out_test, output1)*100
print(acc)

cm = confusion_matrix(out_test, output1)
#tn, fp, fn, tp = cm.ravel()
print('CONFUSION MATRIX ------------------')
print(cm)


# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("./doc/images/" + network + "_plot.png")

##################################################################################################
# plot the CNN model #DB
plot_model(model, to_file="./doc/images/bd_"+network+".png", show_shapes=True)
print("\nTRAINING " + network + " FINISHED\n")



