import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
class Evaluate:
    def __init__(self, stock):
        '''引入資料'''
        day = 15
        self.x_test = np.load('./StockData/TrainingData/NormtestingX_'+stock+'.npy')
        self.y_test = np.load('./StockData/TrainingData/testingY_'+stock+'.npy')
        self.origin_x_test = np.load('./StockData/TrainingData/opentestingX_'+stock+'.npy') #每個禮拜一的開盤價
        self.model = load_model('./stockModel/stockmodel_'+stock+'.h5') #引入訓練完model
        self.x_test = self.x_test.reshape(-1,day,self.x_test.shape[1])
        self.predict = self.model.predict(self.x_test)
        self.stock = stock

    def roi(self,method): #method=> predict ans baseline
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
        for standord, real, open_money in data: #放空 = close - open < 0
            if standord > 0:
                amount = funds/open_money #買幾張
                amount = math.floor(amount)
                funds += (amount * real)
            if standord < 0:
                close_money = standord + open_money
                amount = funds/close_money
                amount = math.floor(amount)
                funds += (amount*(-1*real))
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
        print("next week :",newweek,"\n")

    def predictplt(self):
        plt.plot(self.y_test,color = 'red',label = 'real stock price')
        plt.plot(self.predict , color = 'blue' , label = 'predict stock price')
        plt.xlabel('week')
        plt.ylabel('stock price')
        plt.legend()
        plt.savefig("./predictimg/"+self.stock+".png")
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
    def trend_accurancy_rate(self):
        trend_acc = 0.00
        for predict, real in zip(self.predict, self.y_test):
            if(predict[0] > 0 and real > 0):
                trend_acc+=1
            elif(predict[0] < 0 and real < 0):
                trend_acc+=1
        trend_acc /= self.y_test.size
        trend_acc *= 100
#<<<<<<< Updated upstream
#<<<<<<< Updated upstream
        print("trend_accurancy_rate:", round(trend_acc,2), "%\n")
        return round(trend_acc,2)
'''=======
        print("trend_accurancy_rate:", round(trend_acc,2), "%\n")
>>>>>>> Stashed changes
=======
        print("trend_accurancy_rate:", round(trend_acc,2), "%\n")
>>>>>>> Stashed changes'''
