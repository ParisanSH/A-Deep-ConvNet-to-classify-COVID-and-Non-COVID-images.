
# baseline model with dropout on the cifar10 dataset
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from sklearn import metrics
from PIL import Image
import sys
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt


from os import listdir
from numpy import asarray
import numpy as np

from numpy import argmax
np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array
######################
###      reading train data    #########
X_covid_train=np.zeros((1000,28,28))
i=0
for filename in listdir('COVID_mod_Train/'):

    image = Image.open('COVID_mod_Train/' + filename)  # open colour image

    X_covid_train[i,:,:]=image
    i=i+1


############################
X_NonCovid_train=np.zeros((999,28,28))

np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array
i=0
for filename in listdir('NonCOVID_mod_Train/'):

    image = Image.open('NonCOVID_mod_Train/' + filename)  # open colour image

    X_NonCovid_train[i,:,:]=image
    i=i+1

X_train=np.concatenate((X_covid_train, X_NonCovid_train), axis=0)
Y_covid_train=np.zeros((1000,))
Y_NonCovid_train=np.ones((999,))
Y_train=np.concatenate((Y_covid_train, Y_NonCovid_train), axis=0)
###############################
###        reading test data      ########
X_covid_test=np.zeros((252,28,28))

i=0
for filename in listdir('COVID_mod_Test/'):

    image = Image.open('COVID_mod_Test/' + filename)  # open colour image

    X_covid_test[i,:,:]=image
    i=i+1


############################
X_NonCovid_test=np.zeros((230,28,28))

np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array
i=0
for filename in listdir('NonCOVID_mod_Test/'):

    image = Image.open('NonCOVID_mod_Test/' + filename)  # open colour image

    X_NonCovid_test[i,:,:]=image
    i=i+1

X_test=np.concatenate((X_covid_test, X_NonCovid_test), axis=0)
Y_covid_test=np.zeros((252,))
Y_NonCovid_test=np.ones((230,))
Y_test=np.concatenate((Y_covid_test, Y_NonCovid_test), axis=0)


# load train and test dataset
def load_dataset(X_train,Y_train,X_test,Y_test):
	# load dataset
	#(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
    trainX=X_train
    trainY_main=Y_train
    testX=X_test
    testY_main=Y_test
    trainY = to_categorical(trainY_main)
    testY=to_categorical(testY_main)
    return trainX,trainY,testX,testY,trainY_main,testY_main

# scale pixels
def prep_pixels(trainX, testX):



	# reshape grayscale images to have a single channel
	width, height, channels = trainX.shape[1], trainX.shape[2], 1
	train = trainX.reshape((trainX.shape[0], width, height, channels))
	test = testX.reshape((testX.shape[0], width, height, channels))
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(28, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
	model.add(Conv2D(28, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
    trainX, trainY, testX, testY,trainY_main,testY_main = load_dataset(X_train,Y_train,X_test,Y_test)
    trainX, testX = prep_pixels(trainX, testX)
    print(trainX.shape)
    model = define_model()
    history = model.fit(trainX, trainY, epochs=100, batch_size=64,verbose=0)
    y_predict=model.predict(testX)
    y_predict2=np.zeros(len(testY))
    
    for j in range(len(testY)):
        y_predict2[j]=argmax(y_predict[j])
    print(argmax(y_predict))
    precision,recall,fscore,_=precision_recall_fscore_support(testY_main, y_predict2, average='macro')
    print('precision is: ',' %.3f' % (precision*100))
    print('recall is: ',' %.3f' % (recall*100))
    print('fscore is: ',' %.3f' % (fscore*100))
    ###
    accuracy=accuracy_score(testY_main, y_predict2)
    print('accuracy is: ',' %.3f' % (accuracy*100))
    ###
    fpr, tpr, thresholds = metrics.roc_curve(testY_main, y_predict2)
    plt.plot(fpr, tpr)
    #####
    auc = metrics.auc(fpr, tpr)
    print('auc is: ',' %.3f' % (auc))

run_test_harness()