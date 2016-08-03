"""
This is simple example for a constitutional neural network aimed at Tumur segmentation from MRI images. 

This category of problems fails under the general category of binary classification. 

Architectures that are aimed at schematic segmentation are definitely better options for this task. This is just for demonstration purposes. 


The code was developed utilizing python 2.7.11 and it is based on the keras package (should work with python 3 as well). 

"""

# Import libraries. 

from __future__ import print_function
import os
import sys
"""
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu3,floatX=float32,optimizer=fast_compile'
os.environ['KERAS_BACKEND'] = 'theano'

In case you want to select a graphic card (i the above code i set the 3rd graphic card.) 
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import keras 
import keras.backend as K
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
import math
from keras import callbacks
import glob
from PIL import Image
from sklearn.cross_validation import train_test_split


# It is good to know the pid of the running code in case you need to stop  or monitor. 
print (os.getpid())
file_open = lambda x,y: glob.glob(os.path.join(x,y))

# learning rate schedule. It is helpful when the learning rate can be dynamically set up. We will be using the callback functionality that keras provides. 
def step_decay(epoch):
	initial_lrate =0.01
	drop = 0.3
	epochs_drop = 30.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	print (lrate)
	return lrate

# The following function will be used to give a number of the parameters in our model. Useful when we need to get an estimate of what size of dataset we have to use.  
def size(model): 
	return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

def createmodel(img_channels,img_rows,img_cols,classes=1):
	# This is a Sequential model. Graph models can be used in order to create more complex networks. 
	# Teaching Points:
	# 1. Here we utilize the adam optimization algorithm. In order to use the SGD algorithm one could replace the {adam=keras.optimizers.Adadelta(lr=0)} line with  {sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)} make sure you import the correct optimizer from keras. 
	# 2. This is a binary classification problem so make sure that the correct activation loss function combination is used. For such a problem the sigmoid activation function with the binary cross entropy loss is a good option
	# 3. Since this is a binary problem use 	model.add(Dense(1)) NOT 2...
	# 4. For multi class model this code can be easily modified by selecting the softmax as activation function and the categorical cross entropy as loss 

	model = Sequential()

	# first set of CONV => RELU => POOL
	model.add(Convolution2D(20, 5, 5, border_mode="same",
		input_shape=(img_channels, img_rows, img_cols)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# second set of CONV => RELU => POOL
	model.add(Convolution2D(50, 5, 5, border_mode="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation("relu"))
	# softmax classifier
	model.add(Dense(classes))
	# model.add(Activation('relu'))
	model.add(Activation('sigmoid'))
	# learning schedule callback
	adam=keras.optimizers.Adadelta(lr=0)
	lrate = LearningRateScheduler(step_decay)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	plot(model, to_file='modellen.png')
	return model

def shuffle(X, y):
	perm = np.random.permutation(len(y))
	X = X[perm]
	y = y[perm]
	print (np.shape(X))
	return X, y


def read_data(image):
	"opens image and converts it to a m*n matrix" 
	image = Image.open(image)
	image = image.getdata()
	# image = list(image.getdata())
	# image = map(list,image)
	image = np.array(image)
	return image.reshape(-1)

def createTrainTestValset(image_dir1, image_dir2):
	Class1_images = file_open(image_dir1,"*.jpg")
	Class2_images = file_open(image_dir2,"*.jpg")
	Class1_set = []
	Class2_set = []
	# REad all the files, and create numpy arrays. 
	Class1_set = [read_data(image) for image in Class1_images]
	Class2_set = [read_data(image) for image in Class2_images]
	Class1_set = np.array(Class1_set) #This is where the Memory Error occurs
	Class2_set = np.array(Class2_set)
	X=np.vstack((Class1_set, Class2_set))
	X=X.astype(np.float)/255
	# print (np.shape(X))
	yclass1=np.zeros((np.shape(Class1_set)[0]))
	yclass2=np.ones((np.shape(Class2_set)[0]))
	# print (np.shape(yclass1))
	y=np.concatenate((yclass1, yclass2))
	# print (np.shape(y))	
	X,y=shuffle(X, y)
	print (np.shape(X))	
	print (np.max(X))
	print (np.shape(y))	
	X_train, X_val,y_train, y_val= train_test_split(X,y, test_size=0.2, random_state=42)
	return X_train,y_train, X_val, y_val 

	# Read the images; and split them in three different sets. 
def trainandpredict(Scan=32 ,img_channels=1,batch_size=64,nb_epoch=300,data_augmentation=False):
	img_rows=Scan
	img_cols=Scan
	image_dir1=os.getcwd()+"/negative_images/"
	image_dir2=os.getcwd()+"/positive_images/"
	modeleval=createmodel(img_channels,img_rows,img_cols)
	X_train,y_train, X_val, y_val = createTrainTestValset(image_dir1, image_dir2)
	X_train =X_train.reshape(
		-1,  # number of samples, -1 makes it so that this number is determined automatically
		1,   # 1 color channel, since images are only black and white
		Scan,  # first image dimension (vertical)
		Scan,  # second image dimension (horizontal)
	)
	X_val =X_val.reshape(
		-1,  # number of samples, -1 makes it so that this number is determined automatically
		1,   # 1 color channel, since images are only black and white
		Scan,  # first image dimension (vertical)
		Scan,  # second image dimension (horizontal)
	)
	# Callbacks
	best_model = ModelCheckpoint('Final_lenet_adapt_{epoch:03d}_{val_acc:.2f}.h5', verbose=1, monitor='val_loss',save_best_only=True)
	lrate = LearningRateScheduler(step_decay)

	# Data augmentation is always a good choice
	if not data_augmentation:
		print('Not using data augmentation.')
		# modeleval.fit(X_train, y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_split=0.1,callbacks=[best_model,lrate],shuffle=True)
	else:
		print('Using real-time data augmentation.')
		print ("pending")
		sys.exit()
		# this will do preprocessing and realtime data augmentation
		# datagen = ImageDataGenerator(
		# 	featurewise_center=False,  # set input mean to 0 over the dataset
		# 	samplewise_center=False,  # set each sample mean to 0
		# 	featurewise_std_normalization=False,  # divide inputs by std of the dataset
		# 	samplewise_std_normalization=False,  # divide each input by its std
		# 	zca_whitening=False,  # apply ZCA whitening
		# 	rotation_range=3,  # randomly rotate images in the range (degrees, 0 to 180)
		# 	width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
		# 	height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
		# 	horizontal_flip=True,  # randomly flip images
		# 	vertical_flip=False)  # randomly flip images
		# modeleval.fit(X_train, y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_train1, y_train1),callbacks=[best_model,lrate],shuffle=True)

	modeleval.load_weights('Final_lenet_adapt_181_0.74.h5')
	# Some evaluation Just the basic stuff... 
	print (dir(modeleval))
	Y_cv_pred = modeleval.predict(X_val, batch_size = 32)
	roc =roc_auc_score(y_val, Y_cv_pred)
	print("ROC:", roc)
	print (Y_cv_pred)
	Y_cv_pred[Y_cv_pred>=.5]=1
	Y_cv_pred[Y_cv_pred<.5]=0
	target_names=[] 
	# print ("The f1-score gives you the harmonic mean of precision and recall. The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes.The support is the number of samples of the true response that lie in that class.")
	target_names = ['class 0', 'class 1']
	print(classification_report(y_val, Y_cv_pred, target_names=target_names,digits=4))


	return 0

if __name__ == '__main__':
    trainandpredict()