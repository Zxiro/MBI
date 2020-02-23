class Evaluate:
    def _init_(self,stock):
        '''引入資料'''
        x_test = np.load('./StockData/TrainingData/NormtestingX_stock'+stock+'.npy')
        y_test = np.load('./StockData/TrainingData/testingY_stock'+stock+'.npy')
        origin_x_test = np.load('./StockData/TrainingData/opentestingX_stock'+stock+'.npy') #每個禮拜一的開盤價
        model = load_model('./stockmodel/stockmodel_'+stock+'.h5') #引入訓練完model
    def roi():
        '''
        利用預測的數字如果為正，依照預測的數字比例來進行購買股票以1000為基準乘以(1+預測數字)
        而realmoney則為利用y_test的值去判斷是否依照比例進行購買
        '''
        get_from_predict = 0   #預測所能獲得金額
        predict_cost = 0 #成本
        real_cost = 0
        get_from_real = 0   #真實獲得金額
        amount = 1000  #購買的數量
        predict = model.predict( x_test )
        for predict_money , real_money , open_money in zip( predict , y_test ,origin_x_test):
            if predict_money > 0:
                get_from_predict += int( amount ) * real_money
                predict_cost += open_money * int( amount )
            if real_money > 0 :
                get_from_real += int( amount ) * real_money
                real_cost += open_money*1000
                #print(get_from_real)
                #get_from_predict = round( get_from_predict[0] , 2 )
                #get_from_real = round( get_from_real , 2 )
        print('predict money:' , get_from_predict)
        print('predict cost' , predict_cost)
        print('real money:' , get_from_real)
        money = round((( get_from_predict/predict_cost )*100) , 2)
        print(money , '%')
        money = round((( get_from_real/real_cost )*100) , 2)
        print(money , '%')



