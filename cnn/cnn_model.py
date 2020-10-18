import sys
sys.path.insert(1, '../Evaluate')


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from model_fit import load_data

#binary 判斷漲跌

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#if len(sys.argv) < 2:
    #stock_symbol = input('enter stock number:')
#else:
    #stock_symbol = sys.argv[1]

stock_symbol = "0050"
x_train, y_train, x_test, y_test, x_val, y_val = load_data(stock_symbol)
val_data = (x_val, y_val)
span = x_test.shape[1]
feature = x_test.shape[2]
print(feature)
model = Sequential()

model.add(Conv1D(filters =128,
                kernel_size = 3,
                strides = 1,
                input_shape = (span, feature),
                activation = 'relu',
               # padding = 'same'
                ))

# model.add(MaxPooling1D(2, padding = 'valid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv1D(filters = 64,
                kernel_size = 3,
                strides = 1,
                activation = 'relu',
                padding = 'same'))
model.add(MaxPooling1D(2, padding = 'valid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv1D(filters = 32,
                kernel_size = 3,
                strides = 1,
                activation = 'relu',
                #padding = 'same'
                ))
model.add(tf.keras.layers.BatchNormalization())
# model.add(AveragePooling1D(2, padding = 'valid'))
# model.add(Conv1D(filters = 32,
#                 kernel_size = 2,
#                 strides = 1,
#                 activation = 'relu'))
#
#model.add(MaxPooling1D(2, padding = 'valid'))
#model.add(AveragePooling1D(2, padding = 'valid'))
model.add(Dropout(0.7)) #Solve overfitting -> batch normalization
model.add(Flatten())
model.add(Dense((1), activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
callback = EarlyStopping(monitor="val_accuracy", patience = 32, verbose = 1, mode="auto")

model.fit(x_train, y_train, epochs = 256, batch_size = 10, verbose = 1, validation_data = val_data,  callbacks=[callback])
model.save('../stockModel/stockmodel_cnn_0050_dif.h5')
means = []
