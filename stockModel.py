import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import sys
if len(sys.argv) < 2:
    stock_symbol = input('enter stock number:')
else:
    stock_symbol = sys.argv[1]
x_train = np.load('./StockData/TrainingData/NormtrainingX_'+stock_symbol+'.npy')
y_train = np.load('./StockData/TrainingData/trainingY_'+stock_symbol+'.npy')
x_test = np.load('./StockData/TrainingData/NormtestingX_'+stock_symbol+'.npy')
y_test = np.load('./StockData/TrainingData/testingY_'+stock_symbol+'.npy')
#x_train = np.where(np.isnan(x_train), 0, x_train)
feature = x_train.shape[1]
#y_train = np.where(np.isnan(y_train), 0, y_train)
x_train =x_train.reshape(-1,5,feature)
x_test = x_test.reshape(-1,5,feature)
#x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
#print(x_train[1])
#print(y_train)

model = Sequential()
print(x_train.shape[2])
model.add(LSTM(50,input_shape=(5,x_train.shape[2]),return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences =False))
#model.add(Dropout(0.2))
#model.add(LSTM(10))
#model.add(Dropout(0.2))
model.add(Dense(1))

sgd = optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="mse",optimizer=sgd)

model.summary()

callback = EarlyStopping(monitor="loss", patience=50, verbose=1, mode="auto")
#callback = [logger]
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
model.fit(x_train,y_train, epochs=1250, batch_size=20, callbacks=[callback,tbCallBack],validation_split=0.2)
#model.fit(x_train,y_train, epochs=1000, batch_size=20, callbacks=[callback],validation_split=0.2)

model.save('./stockModel/stockmodel_'+stock_symbol+'.h5')
