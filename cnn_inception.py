import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization,concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#if len(sys.argv) < 2:
    #stock_symbol = input('enter stock number:')
#else:
    #stock_symbol = sys.argv[1]
stock_symbol = "0050"
x_train = np.load('./StockData/TrainingData/NormtrainingX_'+stock_symbol+'.npy')
y_train = np.load('./StockData/TrainingData/trainingY_'+stock_symbol+'.npy')
x_test = np.load('./StockData/TrainingData/NormtestingX_'+stock_symbol+'.npy')
y_test = np.load('./StockData/TrainingData/testingY_'+stock_symbol+'.npy')
y_train_mon = np.load('./StockData/TrainingData/trainingY_mon_'+stock_symbol+'.npy')
y_train_fri = np.load('./StockData/TrainingData/trainingY_fri_'+stock_symbol+'.npy')

'''def turn_to_bin(list):
    for i in range(len(list)):
        if(list[i]>=0):
            list[i] = 1
        else:
            list[i] = 0
   # print(list)

turn_to_bin(y_train)
print(y_train)
turn_to_bin(y_test)
print(y_test)'''
#turn_to_bin(y_train_mon)
#turn_to_bin(y_train_fri)
#exit()

'''print(x_train.shape) #(3065, 21) 每筆都是整周的fea 3065 = 613 (week num) * 5 (day)
print(x_train)
print(y_train.shape)
print(y_train)
print(x_test)'''
print(x_test.shape)
feature = x_test.shape[2]

def layer_1(input_data):

    lay_1 = Conv1D(filters = 128,
                kernel_size = 2,
                strides = 1,
                activation = 'relu')(input_data)

    #(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

    lay_2 = Conv1D(filters = 64,
                    kernel_size = 2,
                    strides = 1,
                    activation = 'relu')(lay_1)
    lay_2 = AveragePooling1D(2, padding = 'valid', strides = 1)(lay_2)

    return lay_2



def inception_module(input_data):
    #input_data = Input(shape = (5, feature))

    branch_1_5 = Conv1D(filters = 128,
                    kernel_size = 1,
                    strides = 1,
                    activation = 'relu')(input_data )
    branch_1_5 = Conv1D(filters = 64,
                    kernel_size = 1,
                    strides = 1,
                    activation = 'relu')(branch_1_5) #output 5

    branch_1_3 = Conv1D(filters = 128,
                    kernel_size = 3,
                    strides = 1,
                    activation = 'relu')(input_data )
    branch_1_3 = Conv1D(filters = 64,
                    kernel_size = 1,
                    strides = 1,
                    activation = 'relu')(branch_1_3) #output 3

    branch_1_1 = Conv1D(filters = 128,
                    kernel_size = 2,
                    strides = 1,
                    activation = 'relu')(input_data )
    branch_1_1 = Conv1D(filters = 64,
                    kernel_size = 2,
                    strides = 1,
                    activation = 'relu')(branch_1_1)
    #branch_1_1 = AveragePooling1D(2, padding = 'valid', strides = 1)(branch_1_1) #output 1

    #bn_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

    output = tf.keras.layers.Concatenate(axis = 1)([branch_1_5, branch_1_3, branch_1_1])

    return output

def cnn(input_data):
    out2 = inception_module(input_data)
    out3 = inception_module(out2)
    out4 = inception_module(out3)
    #out5 = inception_module(out4)
    res = layer_1(out4)
    output = Flatten()(res)
    dropout = tf.keras.layers.Dropout(0.5)(output)
    dense = Dense(1, activation='linear')(dropout)
    model = Model(inputs = input_data, outputs = dense)
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
input_ = Input(shape = (5, feature))
model = cnn(input_)
index = list(range(len(x_train)))
np.random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]
y_train_mon = y_train_mon[index]
y_train_fri = y_train_fri[index]

callback = EarlyStopping(monitor="val_loss", patience = 32, verbose = 1, mode="auto")
model.fit(x_train, y_train_mon, epochs = 512, batch_size = 8, verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('./stockModel/stockmodel_inception_cnn_0050_mon.h5')

model.fit(x_train, y_train_fri, epochs = 512, batch_size = 8, verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('./stockModel/stockmodel_inception_cnn_0050_fri.h5')

model.fit(x_train, y_train, epochs = 512, batch_size = 8, verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('./stockModel/stockmodel_inception_cnn_0050_dif.h5')


