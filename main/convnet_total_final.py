import numpy as np
import cv2
import os
from os.path import isfile,isdir
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, AveragePooling2D,LeakyReLU,ReLU
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score,roc_curve,auc,RocCurveDisplay
import matplotlib.pyplot as plt
# tf.random.set_seed(1)


def load_data(path, label, image_size=(256, 256)):
    '''
    load data
    resize data
    '''
    images = list()
    images_name = os.listdir(path)
    for index, image_name in enumerate(images_name):
        print(f'\rreading {label} data. image : {index + 1} / {len(images_name)}', end='')
        image = cv2.imread(os.path.join(path, image_name))

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)
            images.append(image)

    labels = np.full(shape=len(images), fill_value=label)
    print()
    return images, labels


def one_hot(y):
    '''
    one hot label
    '''
    o_t = np.zeros((len(y), int(max(y)+1)))
    o_t[np.arange(len(y)), y.astype('int32')] = 1
    return o_t


def define_model(num_classes=2,image_size=(256,256,3)):
    '''
    define convnet model model
    '''
    model = Sequential()

    model.add(Conv2D(4, kernel_size=(3, 3),input_shape=image_size))
    model.add(ReLU())
    model.add(AveragePooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.1))


    model.add(Conv2D(16, (3, 3)))
    model.add(ReLU())
    model.add(AveragePooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(LeakyReLU(.1))           
    model.add(Dropout(0.1))

    model.add(Dense(256))
    model.add(LeakyReLU(.1))           
    model.add(Dropout(0.1))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))           
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    return model




def train_model(model,train_data, train_data_label,save_model_path,epochs = 100,batch_size = 100,):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    model.fit(train_data, train_data_label, batch_size=batch_size,epochs=epochs,verbose=1)
    print('saving model...')
    model.save(save_model_path)
    print('train model done.')

    return model



def calculate_metrics(model,test_data, test_data_label,):
    y_predict = model.predict(test_data)

    y_predict = np.argmax(y_predict,axis=1)
    y_true = np.argmax(test_data_label,axis=1)

    print(f'accuracy : {accuracy_score(y_true,y_predict)*100} %')
    print(f'precision : {precision_score(y_true,y_predict)*100} %')
    print(f'recall : {recall_score(y_true,y_predict)*100} %')
    print(f'f1_score : {f1_score(y_true,y_predict)*100} %')
    print(f'auc : {roc_auc_score(y_true,y_predict)} ')
    

    fpr, tpr, thresholds = roc_curve(y_true,y_predict)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='estimator')
    display.plot()  
    plt.savefig('roc.png')
    plt.show()      
    

    
def main():
    image_size=(256,256)
    # tf.random.set_seed(2)

    path_covid_train = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/train/COVID'
    path_non_covid_train = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/train/Non-COVID'
    path_covid_test = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/test/COVID'
    path_non_covid_test = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/test/Non-COVID'

    path_train_data = '/content/drive/MyDrive/nndl/train_data.npy'
    path_train_data_label = '/content/drive/MyDrive/nndl/train_data_label.npy'
    path_test_data = '/content/drive/MyDrive/nndl/test_data.npy'
    path_test_data_label = '/content/drive/MyDrive/nndl/test_data_label.npy'

    # path_covid_train = r'COVID,Non COVID-CT Images\train\COVID'
    # path_non_covid_train = r'COVID,Non COVID-CT Images\train\Non-COVID'
    # path_covid_test = r'COVID,Non COVID-CT Images\test\COVID'
    # path_non_covid_test = r'COVID,Non COVID-CT Images\test\Non-COVID'

    # path_train_data = 'train_data.npy'
    # path_train_data_label = 'train_data_label.npy'
    # path_test_data = 'test_data.npy'
    # path_test_data_label = 'test_data_label.npy'


    if isfile(path_train_data) and isfile(path_train_data_label) and isfile(path_test_data) and isfile(path_test_data_label): 
        f1 = open(path_train_data,'rb')
        f2 = open(path_train_data_label,'rb')
        f3 = open(path_test_data,'rb')
        f4 = open(path_test_data_label,'rb')

        train_data = np.load(f1)
        train_data_label = np.load(f2)
        test_data = np.load(f3)
        test_data_label = np.load(f4)
        print('load data complete.')
    
    else:
        
        print('reading train data...')
        images_covid_train, images_covid_train_labels = load_data(path_covid_train, 'covid',image_size)
        images_non_covid_train, images_non_covid_train_labels = load_data(path_non_covid_train, 'non_covid',image_size)
        print('reading test data...')
        images_covid_test, images_covid_test_labels = load_data(path_covid_test, 'covid',image_size)
        images_non_covid_test, images_non_covid_test_labels = load_data(path_non_covid_test, 'non_covid',image_size)
        
        train_data = np.vstack([images_covid_train, images_non_covid_train])
        train_data_label = np.hstack([images_covid_train_labels, images_non_covid_train_labels])
        
        test_data = np.vstack([images_covid_test, images_non_covid_test])
        test_data_label = np.hstack([images_covid_test_labels, images_non_covid_test_labels])

        train_data_label[train_data_label=='covid'] = 1
        train_data_label[train_data_label=='non_covid'] = 0
        train_data_label = train_data_label.astype('int32')
        
        test_data_label[test_data_label=='covid'] = 1
        test_data_label[test_data_label=='non_covid'] = 0
        test_data_label = test_data_label.astype('int32')
        
        train_data_label = one_hot(train_data_label)
        test_data_label = one_hot(test_data_label)
        
        print('saving data.')
        f1 = open(path_train_data,'wb')
        f2 = open(path_train_data_label,'wb')
        f3 = open(path_test_data,'wb')
        f4 = open(path_test_data_label,'wb')

        np.save(f1,train_data)
        np.save(f2,train_data_label)
        np.save(f3,test_data)
        np.save(f4,test_data_label)

        print('load data complete.')
        
        
    save_model_path = "/content/drive/MyDrive/nndl/model.h5py"
    
    if isdir(save_model_path):
        model = tf.keras.models.load_model(save_model_path)
    
    else:
        epochs = 50
        batch_size = 100
        image_size = image_size + (3,)
        num_classes = len(np.unique(train_data_label))
        model = define_model(num_classes=num_classes,image_size=image_size)
        model = train_model(model,train_data, train_data_label,save_model_path,epochs = epochs,batch_size = batch_size,)
    
    calculate_metrics(model,test_data, test_data_label,)
    
    
    return model



model = main()