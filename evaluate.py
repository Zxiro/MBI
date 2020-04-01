import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
class Evaluate:
    def __init__(self,stock):
        '''引入資料'''
        self.x_test = np.load('./StockData/TrainingData/NormtestingX_stock'+stock+'.npy')
        self.y_test = np.load('./StockData/TrainingData/testingY_stock'+stock+'.npy')
        self.origin_x_test = np.load('./StockData/TrainingData/opentestingX_stock'+stock+'.npy') #每個禮拜一的開盤價
        self.model = load_model('./stockModel/stockmodel_'+stock+'.h5') #引入訓練完model
        self.x_test = self.x_test.reshape(-1,5,self.x_test.shape[1])
        self.predict = self.model.predict(self.x_test)
        self.stock = stock
    def roi(self):
        principle = 1000000 # 本金
        predict_money = 1000000
        ans_money = 1000000
        amount = 0
        for predict, real, open_money in zip(self.predict, self.y_test, self.origin_x_test):
            if predict > 0:
                amount = predict_money/open_money
                amount = math.floor(amount)
                predict_money += (amount * real)
            print(predict)
            print(real)
            print(type(real))
            if real > 0:
                amount = ans_money/open_money
                amount = math.floor(amount)
                ans_money += (amount * real)
        predict_money = round(predict_money,2)
        print("predict_money: ",round(predict_money,2))
        print("roi of predict: ",round((predict_money-principle)/principle*100,2),"%")
        print("ans_money: ",round(ans_money,2))
        print("roi of ans: ", round((ans_money-principle)/principle*100,2),"%\n")
    def stable_roi(self):
        '''
        利用預測的數字如果為正，依照預測的數字比例來進行購買股票以1000為基準乘以(1+預測數字)
        而realmoney則為利用y_test的值去判斷是否依照比例進行購買
        '''
        #print(self.x_test)
        get_from_predict = 0   #預測所能獲得金額
        predict_cost = 0 #成本
        real_cost = 0
        get_from_real = 0   #真實獲得金額
        amount = 1000  #購買的數量
        predict = self.predict
        for predict_money , real_money , open_money in zip(predict , self.y_test ,self.origin_x_test):
            if predict_money > 0:
                get_from_predict += amount  * real_money
                predict_cost += open_money * amount
            if real_money > 0 :
                get_from_real += amount * real_money
                real_cost += open_money*1000
                #print(get_from_real)
                #get_from_predict = round( get_from_predict[0] , 2 )
                #get_from_real = round( get_from_real , 2 )
        print('predict money:', get_from_predict)
        print('predict cost', predict_cost)
        print('real money:', get_from_real)
        money = round((( get_from_predict/predict_cost )*100) , 2)
        print('roi with predict:', money, '%')
        money = round((( get_from_real/real_cost )*100) , 2)
        print('roi with answer:',money , '%\n')

    def nextweek_predict(self):
        newweek = self.predict[-1]
        newweek = round(newweek[0] , 2)
        print("next week :",newweek,"\n")

    def predictplt(self):
        plt.plot(self.y_test,color = 'red',label = 'real stock price')
        plt.plot(self.predict , color = 'blue' , label = 'predict stock price')
        plt.xlabel('time')
        plt.ylabel('stock price')
        plt.legend()
        plt.savefig("./predictimg/"+self.stock+".png")
        plt.show()

    def baseline(self):
        amount = 0
        total = 1000000
        principle = 1000000
        for cost, last_price, real in zip(self.origin_x_test[1:], self.y_test[:-2], self.y_test[1:]):
            if(last_price > 0):
                amount = math.floor(total/cost)
                total += (real * amount)
        print("principle: ",principle)
        print("baseline earn: ",round(total - principle,2))
        money = round((((total-principle)/principle )*100), 2)
        print("roi for baselinue: ", money, "%\n")

    '''預測誤差的百分比'''
    def accurancy_rate(self):
        acc_rate = 0.00;
        zero_acount = 0;
        for predict, real  in zip(self.predict, self.y_test):
            if(real == 0):
                zero_acount += 1
                continue
            acc_rate += abs(((predict[0]-real)/real))

        acc_rate /= (self.y_test.size)
        acc_rate *= 100
        print(type(acc_rate))
        print("accurate rate: ", round(acc_rate,2), "%\n")

    '''預測正負的準確度'''
    def trend_accurancy_rate(self):
        trend_acc = 0.00
        for predict, real in zip(self.predict[0], self.y_test):
            if(predict>0 and real>0):
                trend_acc+=1
            elif(predict<0 and real<0):
                trend_acc+=1
        trend_acc /= self.y_test.size
        trend_acc *= 100
        print("trend_accurancy_rate:", round(trend_acc,2), "%\n")
