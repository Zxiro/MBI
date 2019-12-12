from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt




# data = pd.read_csv('./StockData/TrainingData/stock0050TrainingData.csv')


x_train = np.load('./StockData/TrainingData/trainingX_stock0050.npy')
y_train = np.load('./StockData/TrainingData/trainingY_stock0050.npy')

x_train = np.where(np.isnan(x_train), 0, x_train)
y_train = np.where(np.isnan(y_train), 0, y_train)
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.33,random_state=42)

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

model.add(LSTM(50,input_shape=(5,5),return_sequences = True))

model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences =True))
model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(100))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss="mse",optimizer="adam")
model.summary()
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(x_train,y_train, epochs=1000, batch_size=20, callbacks=[callback],validation_data=(x_train,y_train),validation_split=0.2)
