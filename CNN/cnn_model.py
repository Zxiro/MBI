import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

#分類binary 判斷漲跌

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#if len(sys.argv) < 2:
    #stock_symbol = input('enter stock number:')
#else:
    #stock_symbol = sys.argv[1]

stock_symbol = "0050"
x_train = np.load('../stock_data/trx/train_x_'+stock_symbol+'.npy')
y_train = np.load('../stock_data/try/train_y_'+stock_symbol+'.npy')
x_test = np.load('../stock_data/tex/test_x_'+stock_symbol+'.npy')
# y_train_mon = np.load('../stock_data/TrainingData/trainingY_mon_'+stock_symbol+'.npy') #將插值轉乘01
# y_train_fri = np.load('../stock_data/TrainingData/trainingY_fri_'+stock_symbol+'.npy')

feature = x_test.shape[2]
model = Sequential()

'''model.add(Conv1D(filters = 512,
                kernel_size = 2,
                strides = 1,
                input_shape = (5, feature),
                activation = 'relu'))'''

model.add(Conv1D(filters = 256,
                kernel_size = 2,
                strides = 1,
                input_shape = (3, feature),
                activation = 'relu'))

#model.add(MaxPooling1D(2, padding = 'valid'))
#model.add(AveragePooling1D(2, padding = 'valid'))

model.add(Conv1D(filters = 128,
                kernel_size = 2,
                strides = 1,
                activation = 'relu'))

#model.add(MaxPooling1D(2, padding = 'valid'))
#model.add(AveragePooling1D(2, padding = 'valid'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense((1), activation='linear'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
callback = EarlyStopping(monitor="val_loss", patience = 32, verbose = 1, mode="auto")

'''index = list(range(len(x_train)))
np.random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]
y_train_mon = y_train_mon[index]
y_train_fri = y_train_fri[index]'''

#model.fit(x_train, y_train_mon, epochs = 512, batch_size = 10, verbose = 1, validation_split = 0.1,  callbacks=[callback])
#model.save('../stockModel/stockmodel_cnn_0050_mon.h5')

#model.fit(x_train, y_train_fri, epochs = 512, batch_size = 10, verbose = 1, validation_split = 0.1,  callbacks=[callback])
#model.save('../stockModel/stockmodel_cnn_0050_fri.h5')

model.fit(x_train, y_train, epochs = 256, batch_size = 4 , verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('../stockModel/stockmodel_cnn_0050_dif.h5')
