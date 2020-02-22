import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

x_train = np.load('./StockData/TrainingData/NormtrainingX_stock0050.npy')
y_train = np.load('./StockData/TrainingData/trainingY_stock0050.npy')
x_test = np.load('./StockData/TrainingData/NormtestingX_stock0050.npy')
y_test = np.load('./StockData/TrainingData/testingY_stock0050.npy')
feature = x_train.shape[1]
x_train = np.where(np.isnan(x_train), 0, x_train)
x_test = np.where(np.isnan(x_test), 0, x_test)
y_train = np.where(np.isnan(y_train), 0, y_train)
x_train = x_train.reshape(-1,5,feature)
x_test = x_test.reshape(-1,5,feature)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

model = Sequential()
model.add(LSTM(35,input_shape=(5,x_train.shape[2]),return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(35,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(35,return_sequences =False))
model.add(Dense(1))

sgd = optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="mse",optimizer=sgd,metrics=['accuracy'])

model.summary()

callback = EarlyStopping(monitor="loss", patience=80, verbose=2, mode="auto")
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
model.fit(x_train,y_train, epochs=1500, batch_size=20, callbacks=[callback,tbCallBack],validation_data=(x_train,y_train)
,validation_split=0.2)

model.save('my_model.h5') 

model.predict(x_test,verbose=1)

predicted_stock_price = model.predict(x_test)
plt.plot(y_test, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
