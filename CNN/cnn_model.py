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
span = x_test.shape[1]
feature = x_test.shape[2]
model = Sequential()
print(x_train.shape)
exit()
model.add(Conv1D(filters = 128,
                kernel_size = 2,
                strides = 1,
                input_shape = (span, feature),
                activation = 'relu'))

model.add(AveragePooling1D(2, padding = 'valid'))

model.add(Conv1D(filters = 64,
                kernel_size = 2,
                strides = 1,
                #input_shape = (span, feature),
                activation = 'relu'))

#model.add(AveragePooling1D(2, padding = 'valid'))
#model.add(MaxPooling1D(2, padding = 'valid'))
# model.add(Conv1D(filters = 128,
#                 kernel_size = 2,
#                 strides = 1,
#                 activation = 'relu'))
#
#model.add(MaxPooling1D(2, padding = 'valid'))
#model.add(AveragePooling1D(2, padding = 'valid'))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense((1), activation='linear'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
callback = EarlyStopping(monitor="val_loss", patience = 32, verbose = 1, mode="auto")

model.fit(x_train, y_train, epochs = 256, batch_size = 12, verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('../stockModel/stockmodel_cnn_0050_dif.h5')
