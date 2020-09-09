import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping


### 直接做/兩個鄉檢

#分類binary 判斷漲跌

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#if len(sys.argv) < 2:
    #stock_symbol = input('enter stock number:')
#else:
    #stock_symbol = sys.argv[1]

stock_symbol = "0050"
x_train = np.load('../StockData/TrainingData/NormtrainingX_'+stock_symbol+'.npy')
y_train = np.load('../StockData/TrainingData/trainingY_'+stock_symbol+'.npy')
x_test = np.load('../StockData/TrainingData/NormtestingX_'+stock_symbol+'.npy')
y_train_mon = np.load('../StockData/TrainingData/trainingY_mon_'+stock_symbol+'.npy') #將插值轉乘01
y_train_fri = np.load('../StockData/TrainingData/trainingY_fri_'+stock_symbol+'.npy')

print(x_train.shape) #(3065, 21) 每筆都是整周的fea 3065 = 613 (week num) * 5 (day)
print(x_train)
print(y_train.shape)
print(y_train)
feature = x_test.shape[2]


model = Sequential()

model.add(Conv1D(filters = 256,
                kernel_size = 2,
                strides = 1,
                input_shape = (5, feature),
                activation = 'relu'))

model.add(Conv1D(filters = 128,
                kernel_size = 2,
                strides = 1,
                activation = 'relu'))

model.add(Conv1D(filters = 64,
                kernel_size = 2,
                strides = 1,
                activation = 'relu'))

#model.add(MaxPooling1D(3, padding = 'valid'))
#model.add(AveragePooling1D(2, padding = 'valid', strides = 1))
model.add(Dropout(0.45))
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

model.fit(x_train, y_train_mon, epochs = 512, batch_size = 8, verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('../stockModel/stockmodel_cnn_0050_mon.h5')

model.fit(x_train, y_train_fri, epochs = 512, batch_size = 8, verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('../stockModel/stockmodel_cnn_0050_fri.h5')

model.fit(x_train, y_train, epochs = 512, batch_size = 8, verbose = 1, validation_split = 0.15,  callbacks=[callback])
model.save('../stockModel/stockmodel_cnn_0050_dif.h5')