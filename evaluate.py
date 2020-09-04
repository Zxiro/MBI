import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
from Model.transformer import TokenAndPositionEmbedding, TransformerBlock, MultiHeadSelfAttention

class Evaluate:
    def __init__(self, stock):
        '''引入資料'''
        day = 5
        self.x_test = np.load('./StockData/TrainingData/NormtestingX_'+stock+'.npy')
        self.y_test = np.load('./StockData/TrainingData/testingY_'+stock+'.npy')
        self.x_train = np.load('./StockData/TrainingData/trainX_'+stock+'.npy')
        self.y_train = np.load('./StockData/TrainingData/trainY_'+stock+'.npy')
        self.x_val = np.load('./StockData/TrainingData/valX_'+stock+'.npy')
        self.y_val = np.load('./StockData/TrainingData/valY_'+stock+'.npy')
        print(self.x_test.shape)
        self.origin_x_test = np.load('./StockData/TrainingData/opentestingX_'+stock+'.npy') #每個禮拜一的開盤價
        self.model_inc = load_model('./stockModel/stockmodel_inception_cnn_0050_dif.h5') #引入訓練完model
        self.model_cnn = load_model('./stockModel/stockmodel_cnn_0050_dif.h5') #引入訓練完model
        self.model_lstm = load_model('./stockModel/stockmodel_inception_cnn_0050_dif.h5') #引入訓練完model
        #self.model = load_model('./stockModel/stockmodel_'+stock+'.h5') #引入訓練完model
        self.model = load_model('./stockModel/transformer_'+stock+'.h5',custom_objects={'MultiHeadSelfAttention':MultiHeadSelfAttention,'TokenAndPositionEmbedding':TokenAndPositionEmbedding,'TransformerBlock':TransformerBlock,}) #引入訓練完model
        #reloaded_model = tf.keras.experimental.load_from_saved_model('./stockModel/transformer_'+stock+'.h5', custom_objects={'KerasLayer':hub.KerasLayer})
        self.x_test = self.x_test.reshape(-1,day,self.x_test.shape[-1])
        self.x_train = self.x_train.reshape(-1,day,self.x_train.shape[-1])
        self.x_val = self.x_val.reshape(-1,day,self.x_val.shape[-1])
        print(self.x_test.shape)
        #exit()
        self.predict = self.model.predict(self.x_test)
        self.train_predict = self.model.predict(self.x_train)
        self.val_predict = self.model.predict(self.x_val)
        self.stock = stock
    def roi(self, method): #method=> predict ans baseline
        principle = 1000000 #本金
        funds = 1000000 #所有的財產
        amount = 0
        if(method == "predict"):
            data = zip(self.predict, self.y_test, self.origin_x_test)
        elif(method == "ans"):
            data = zip(self.y_test, self.y_test, self.origin_x_test)
        elif(method == "baseline"):
            data = zip(self.y_test[:-2], self.y_test[1:], self.origin_x_test[1:])
        else:
            print("ERROR!!!no this method")
            return
        for predict, real, open_money in data: #放空 = close - open < 0
            if predict > 0:
               # print(predict)
                amount = funds/open_money #買幾張
                amount = math.floor(amount)
                funds += (amount * real)
            if predict < 0:
                close_money = predict + open_money
                amount = funds/close_money
                amount = math.floor(amount)
                funds += (amount * (-1 * real))

        funds = round(funds, 2)
        print(method)
        print("principle: ", principle)
        print("funds: ", round(funds, 2))
        print("income: ", round(funds-principle, 2))
        print("roi of ", method, ": ", round((funds-principle)/principle*100, 2), "%\n")
        return round((funds-principle)/principle*100,2)


    def nextweek_predict(self):
        newweek = self.predict[-1]
        newweek = round(newweek[0] , 2)
        print("next week? :",newweek,"\n")

    def predictplt(self):
        plt.cla()
        plt.plot(self.y_test,color = 'red',label = 'real stock price')
        plt.plot(self.predict , color = 'blue' , label = 'predict stock price')
        plt.xlabel('week')
        plt.ylabel('stock price')
        plt.legend()
        plt.savefig("../public_html/png/tranformer_"+self.stock+"_"+time.strftime("%m-%d-%H-%M%S",time.localtime())+".png")
        plt.show()

    '''預測誤差的百分比'''
    def accurancy_rate(self):
        acc_rate = 0.00
        zero_acount = 0
        for predict, real  in zip(self.predict, self.y_test):
            if(real == 0):
                zero_acount += 1
                continue
            acc_rate += abs(((predict[0]-real)/real))

        acc_rate /= (self.y_test.size)
        acc_rate *= 100
        print("loss rate: ", round(acc_rate,2), "%\n")

    '''預測正負的準確度'''
    def trend_accurancy_rate(self,trend_type):
        trend_acc = 0.00
        if(trend_type == "test"):
            predict_data = self.predict
            y_data = self.y_test
        elif(trend_type == "train"):
            predict_data = self.train_predict
            y_data = self.y_train
        elif(trend_type == "val"):
            predict_data = self.val_predict
            y_data = self.y_val
        else :
            return
        for predict, real in zip(predict_data, y_data):
            if(predict[0] > 0 and real > 0):
                trend_acc+=1
            elif(predict[0] < 0 and real < 0):
                trend_acc+=1
        trend_acc /= y_data.size
        trend_acc *= 100
        print("trend_accurancy_rate:", round(trend_acc,2), "%\n")
        return round(trend_acc,2)

