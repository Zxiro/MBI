import datetime #convert data to datetime64(ns)
import os
import numpy as np
import pandas as pd
from statistics import mean
from sklearn import preprocessing
from build_config import index_dic
from build_config import stock_dic
from add_feature import Add_feature
from get_usa_data import get_usa_index
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

def resha(x): #從 (week, day, features) reshape 成 (week*day, features) -> (total days, features)
    nptrain = np.array(x)
    print(nptrain.shape)
    nptrain = np.reshape(nptrain,(nptrain.shape[0] * nptrain.shape[1], nptrain.shape[2]))
    return nptrain

def standardlize(y):
    y_test = y[-50:] 
    #y = preprocessing.scale(y) # Z Score Standardlization (X-(mean fo x ))/(Standard deviation)
    train_y = y[:-50]
    return train_y, y_test

#tr/ te seperate normalized record normalized tr mean and deviation
#修改shift 也要預測未滿5天的插植
#no normalize y 


def save_np(x, y, open_money, num, mon, fri):
    scale = preprocessing.StandardScaler()
    train_x = x[:-50]
    x_test = x[-50:] #後50筆
    scale = scale.fit(resha(train_x))  #  標準化後的標準scale, resha(x) = two dim /  (tr + te) 

    x_test = scale.transform(resha(x_test)) # two dim standradlized
    train_x = scale.transform(resha(train_x)) # two dim standradlized

    x_test = x_test.reshape((int(x_test.shape[0]/5), 5, -1)) # return to three dim
    train_x  = train_x .reshape((int(train_x .shape[0]/5), 5, -1)) # return to three dim

    train_y, y_test = standardlize(y)
    train_y_mon, y_test_mon = standardlize(mon)
    train_y_fri, y_test_fri = standardlize(fri)

    open_money = open_money[-50:]
    stock_name = num

    Npdata = train_x
    np.save(os.path.join('./StockData/TrainingData/', 'NormtrainingX_' + stock_name), Npdata)
    print(stock_name)
    print(num ," trainX  ", Npdata.shape)
    print(Npdata)

    Npdata = x_test
    np.save(os.path.join('./StockData/TrainingData/', 'NormtestingX_' + stock_name), Npdata)
    print(num, " testX  ", Npdata.shape)
    print(Npdata)

    Npdata = np.array(train_y)
    np.save(os.path.join('./StockData/TrainingData/', 'trainingY_' + stock_name), Npdata)
    print(num, " trainY  ", Npdata.shape)
    print(Npdata)

    Npdata = np.array(y_test)
    np.save(os.path.join('./StockData/TrainingData/', 'testingY_' + stock_name), Npdata)
    print(num, " testY  ", Npdata.shape)
    print(Npdata)

    Npdata = np.array(train_y_mon)
    np.save(os.path.join('./StockData/TrainingData/', 'trainingY_mon_' + stock_name), Npdata)
    print(num, " trainY_mon  ", Npdata.shape)
    print(Npdata)

    Npdata = np.array(y_test_mon)
    np.save(os.path.join('./StockData/TrainingData/', 'testingY_mon_' + stock_name), Npdata)
    print(num, " testY_mon  ", Npdata.shape)
    print(Npdata)

    Npdata = np.array(train_y_fri)
    np.save(os.path.join('./StockData/TrainingData/', 'trainingY_fri_' + stock_name), Npdata)
    print(num, " trainY_fri  ", Npdata.shape)
    print(Npdata)

    Npdata = np.array(y_test_fri)
    np.save(os.path.join('./StockData/TrainingData/', 'testingY_fri_' + stock_name), Npdata)
    print(num, " testY_fri  ", Npdata.shape)
    print(Npdata)


    Npdata = np.array(open_money)
    np.save(os.path.join('./StockData/TrainingData/', 'opentestingX_' + stock_name), Npdata)
    #print(num, " opentestX  ", Npdata.shape)
    #print(Npdata)

def generate_train(feature, data, name):
    train_x = []
    train_y = []
    train_y_mon = []
    train_y_fri = []
    open_money = []
    for _, span_data in data:
        #預測未滿5天
        if len(span_data) == 5: #Decide the way seperate the stock data
            train_x.append(span_data.values.tolist()) #append all feature list
            mon_open = span_data['open'][0]
            fri_close = span_data['close'][4]
            open_money.append(mon_open)
            train_y.append(fri_close - mon_open)
            train_y_fri.append(fri_close)
            train_y_mon.append(mon_open)
        
    train_x.pop()#drop last
    train_y.pop(0)#drop first
    train_y_fri.pop(0)
    train_y_mon.pop(0)
    save_np(train_x, train_y, open_money, name, train_y_mon, train_y_fri)

def filter_feature(df, feature):
    df = df[df.columns[df.columns.isin(feature)]] #篩選出需要的feature
    print(df)
    return df

def load_csv(num, start, end):
    stock_data = pd.DataFrame(pd.read_csv('./StockData/stock'+num+'.csv'))
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    print(stock_data["date"][0])
    count=0; 
    for i in stock_data["date"]:
        if( start_date > i or end_date < i):
            stock_data.drop([count],axis = 0,inplace = True)
        count =count + 1
    stock_data = stock_data.reset_index(drop=True)
    return stock_data
if '__main__' == __name__:
    stock_num = stock_dic['stock_num']
    feature = stock_dic['features']
    span = stock_dic['span']
    close_type = stock_dic['close_type']
    start_date = stock_dic['date']
    end_date = stock_dic['end_date']
    usa = get_usa_index() #get usa index data
    stock_data = load_csv(stock_num, start_date, end_date) #個股加上USA index 全部資料
    af = Add_feature(stock_data)
    #print(feature)
    af.data = filter_feature(af.data, feature) #在該個股上留下需要的feature
    df = pd.concat([af.data, usa], axis=1).reindex(af.data.index) #將美國指數concat到個股上
    df = df.dropna()
    print(df)
    df.to_csv('./train.csv')
    df = df.set_index('date').resample('w')
    generate_train(feature, df, stock_num)