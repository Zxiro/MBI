import os
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from Model.transformer import TokenAndPositionEmbedding, TransformerBlock, MultiHeadSelfAttention
from sklearn.model_selection import train_test_split
def load_data(stock_symbol):
    day = 5
    x_train = np.load('./StockData/TrainingData/NormtrainingX_'+stock_symbol+'.npy')
    y_train = np.load('./StockData/TrainingData/trainingY_'+stock_symbol+'.npy')
    x_test = np.load('./StockData/TrainingData/NormtestingX_'+stock_symbol+'.npy')
    y_test = np.load('./StockData/TrainingData/testingY_'+stock_symbol+'.npy')
    x_train = np.where(np.isnan(x_train), 0, x_train)
    feature = x_train.shape[-1]
    x_train =x_train.reshape(-1,day,feature)
    x_test = x_test.reshape(-1,day,feature)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    '''index = list(range(len(x_train)))

    np.random.shuffle(index)

    print(index)

    x_train = x_train[index]
    y_train = y_train[index]
    '''
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    Npdata = x_train
    np.save(os.path.join('./StockData/TrainingData/', 'trainX_' + stock_symbol), Npdata) 

    Npdata = y_train
    np.save(os.path.join('./StockData/TrainingData/', 'trainY_' + stock_symbol), Npdata)

    Npdata = x_val
    np.save(os.path.join('./StockData/TrainingData/', 'valX_' + stock_symbol), Npdata)

    Npdata = y_val
    np.save(os.path.join('./StockData/TrainingData/', 'valY_' + stock_symbol), Npdata)

    #return x_train,y_train,x_test,y_test
    return x_train, y_train, x_test, y_test, x_val, y_val

def model_fit(model, x_train, y_train, x_test, y_test, x_val, y_val, stock_symbol, model_type):

    sgd = optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss="mse",optimizer=sgd)

    model.summary()

    callback = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="auto")#callback = [logger]

    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                  batch_size=32,     # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=True, # 是否可视化梯度直方图
                     write_images=True,# 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)





    #model.fit(x_train,y_train, epochs=1250, batch_size=20, callbacks=[callback,tbCallBack], validation_split = 0.15)
    model.fit(x_train,y_train, epochs=1250, batch_size=20, callbacks=[callback,tbCallBack], validation_data = (x_val,y_val),)
    model.save('./stockModel/'+model_type+'_'+stock_symbol+'.h5')
