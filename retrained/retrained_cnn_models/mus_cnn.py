#!/usr/bin/python

import random
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cPickle as cP
from sys import argv

from keras import backend as K
K.set_image_dim_ordering('th')

np.random.seed(123)  # for reproducibility
batch_size = 32
nb_classes = 2
nb_epoch = int(argv[1])
nb_filters = 64
seql = 101

def load_data(ipath):
    print 'loading data......'
    fin = open(ipath, 'rb')
    datas = cP.load(fin)

    X_train = np.array([datas[0][i][0] for i in range(len(datas[0]))])
    y_train = np.array([datas[0][i][1] for i in range(len(datas[0]))])
    
    X_cv = np.array([datas[1][i][0] for i in range(len(datas[1]))])
    y_cv = np.array([datas[1][i][1] for i in range(len(datas[1]))])

    X_btest = np.array([datas[2][i][0] for i in range(len(datas[2]))])
    y_btest = np.array([datas[2][i][1] for i in range(len(datas[2]))])

    X_nbtest5 = np.array([datas[3][i][0] for i in range(len(datas[3]))])
    y_nbtest5 = np.array([datas[3][i][1] for i in range(len(datas[3]))])

    X_nbtest10 = np.array([datas[4][i][0] for i in range(len(datas[4]))])
    y_nbtest10 = np.array([datas[4][i][1] for i in range(len(datas[4]))])

    X_nbtest20 = np.array([datas[5][i][0] for i in range(len(datas[5]))])
    y_nbtest20 = np.array([datas[5][i][1] for i in range(len(datas[5]))])

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_cv = np_utils.to_categorical(y_cv, nb_classes)
    Y_btest = np_utils.to_categorical(y_btest, nb_classes)
    Y_nbtest5 = np_utils.to_categorical(y_nbtest5, nb_classes)
    Y_nbtest10 = np_utils.to_categorical(y_nbtest10, nb_classes)
    Y_nbtest20 = np_utils.to_categorical(y_nbtest20, nb_classes)
    
    print 'X_train shape:', X_train.shape
    print 'X_cv shape:', X_cv.shape
    print 'X_btest shape:', X_btest.shape
    print 'X_nbtest5 shape:', X_nbtest5.shape
    print 'X_nbtest10 shape:', X_nbtest10.shape
    print 'X_nbtest20 shape:', X_nbtest20.shape

    print 'Numbers of positive training samples:', y_train.tolist().count(1.)
    
    print 'data loaded success!'
    
    fin.close()
    
    return X_train, Y_train, X_cv, Y_cv, X_btest, Y_btest, X_nbtest5, Y_nbtest5, X_nbtest10, Y_nbtest10, X_nbtest20, Y_nbtest20

def create_model():
    
    model = Sequential()

    #First convolutional layer
    model.add(Convolution2D(nb_filters, 4, 8, border_mode='valid', input_shape=(1, 4, seql), init='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())

    #First pooling layer
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.3))

    #Second convolutional layer
    model.add(Convolution2D(nb_filters/2, 1, 8, border_mode='valid', input_shape=(1, 4, seql), init='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())

    #Second pooling layer
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.3))

    #flatten
    model.add(Flatten())

    #full connected layer
    model.add(Dense(64, init='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    #full connected layer
    model.add(Dense(64, init='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    #output layer
    model.add(Dense(nb_classes, init='he_normal'))
    model.add(Activation('softmax'))

    #sgd = SGD(lr=0.02, momentum=0.92, decay=1e-6, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model

def training(X_train, Y_train, X_cv, Y_cv, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_cv, Y_cv), callbacks=[early_stopping]) # validation_data=(X_cv, Y_cv),
    print 'Model trained!'
    return model

def testing(test_name, X, Y, model):
    score = model.evaluate(X, Y, verbose=0)
    
    print test_name+' loss:', score[0]
    print test_name+' acc:', score[1]
    
    Y_ = model.predict(X)
    print test_name+' test auc:', roc_auc_score(Y, Y_)
    print test_name+' test aupr:', average_precision_score(Y, Y_)
    
    output = open('./results/training/mus_101_'+test_name+'.pkl', 'wb')
    cP.dump([Y, Y_], output)

def save_model(model):
    #json_string = model.to_json()
    #open('cnn_structure.json', 'w').write(json_string)
    model.save_weights('./cnn_weights/cnn_mus_weights.h5')

if __name__ == '__main__':
    ipath = './packed_data/mus_101_all_folds.pkl'
    
    X_train, Y_train, X_cv, Y_cv, X_btest, Y_btest, X_nbtest5, Y_nbtest5, X_nbtest10, Y_nbtest10, X_nbtest20, Y_nbtest20 = load_data(ipath)
    
    cnn_model = create_model()
    trained_model = training(X_train, Y_train, X_cv, Y_cv, cnn_model)
    
    testing('btest', X_btest, Y_btest, trained_model)
    testing('nbtest5', X_nbtest5, Y_nbtest5, trained_model)
    testing('nbtest10', X_nbtest10, Y_nbtest10, trained_model)
    testing('nbtest20', X_nbtest20, Y_nbtest20, trained_model)
    
    save_model(trained_model)
