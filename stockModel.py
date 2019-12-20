from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt




# data = pd.read_csv('./StockData/TrainingData/stock0050TrainingData.csv')


x_train = np.load('./StockData/TrainingData/NormtrainingX_stock0050.npy')
y_train = np.load('./StockData/TrainingData/trainingY_stock0050.npy')

x_train = np.where(np.isnan(x_train), 0, x_train)
y_train = np.where(np.isnan(y_train), 0, y_train)
#x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.33,random_state=42)
x_train =x_train.reshape(671,5,5)
print(x_train.shape)
print(y_train.shape)
'''x_train = []
print(data)
for i in data:
    x_train.append(i)
y_train = data.iloc[0].to_list()
x_train = np.array(x_train)
y_train = np.array(y_train)
'''
#print(x_train[1])

#print(y_train)


model = Sequential()

model.add(LSTM(10,input_shape=(5,5),return_sequences = True))

model.add(Dropout(0.2))
model.add(LSTM(10,return_sequences =False))
model.add(Dropout(0.2))
#model.add(LSTM(10,return_sequences = True))
#model.add(Dropout(0.2))
#model.add(LSTM(10))
#model.add(Dropout(0.2))
model.add(Dense(1))

sgd = optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="mse",optimizer=sgd)

model.summary()
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
model.fit(x_train,y_train, epochs=1000, batch_size=20, callbacks=[callback,tbCallBack],validation_data=(x_train,y_train),validation_split=0.2)
