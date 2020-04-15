import datetime #convert data to datetime64(ns)
import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from sklearn import preprocessing
from build_config import index_dic
from build_config import stock_dic
from talib import abstract

def add_MA(data):
    for ma in ['5', '20', '30', '60']:
        data["MA"+ ma] = data.close.rolling(int(ma)).mean()

def add_MACD(data):
    tmp_df= abstract.MACD(data)
    data['MACD'] = tmp_df[tmp_df.columns[0]].values.tolist()
    data['MACDsignal'] = tmp_df[tmp_df.columns[1]].values.tolist()
    data['MACDhist'] = tmp_df[tmp_df.columns[2]].values.tolist()
    
def add_rsv(data): # rsv (今天收盤-最近9天的最低價)/(最近9天的最高價-最近9天的最低價)
    rsv=[]
    close=data.loc[ : ,'close'].values.tolist() 
    for i in range(0,len(close)):
        if i>=8:
            low = min(data.loc[ i-8 : i ,'low'].values.tolist())
            high = max(data.loc[ i-8 : i ,'high'].values.tolist())
            rsv.append(((close[i]-low)/(high-low))*100)
        else:
            rsv.append(0)
    data['rsv'] = rsv

def add_k(data):  # k (2/3昨日K 加 1/3 今日rsv)
    k=[]
    rsv=data.loc[ : ,'rsv'].values.tolist() 
    for i in range(0,len(rsv)):
        if i>=1:
            k.append(((2/3)*k[i-1])+((1/3)*rsv[i]))
        else:
            k.append(rsv[0])  
    data['k'] = k

def add_d(data): # d (2/3昨日d 加 1/3 今日k)
    d=[]
    k=data.loc[ : ,'k'].values.tolist() 
    for i in range(0,len(k)):
        if i>=1:
            d.append(((2/3)*d[i-1])+((1/3)*k[i]))
        else:
            d.append(k[0])
    data['d'] = d

def resha(x): #從 (幾周,每周幾天,特徵數)reshape成(幾周*一周天數,特徵數) ->(總天數,特徵數)
    nptrain = np.array(x)
    print(nptrain.shape)
    nptrain = np.reshape(nptrain,(nptrain.shape[0] * nptrain.shape[1], nptrain.shape[2]))
    return nptrain

def save_np(x, y, open_money, num):
    train_x = x[:-50]
    train_y = y[:-50]
    x_test = x[-50:] #後50筆
    y_test = y[-50:]
    open_money = open_money[-50:]
    stock_name = num
    scaler = preprocessing.StandardScaler() #初始化scaler
    scaler = scaler.fit(resha(train_x))  #  標準化後的數據
    train_x = scaler.transform(resha(train_x))

    Npdata = train_x
    np.save(os.path.join('./StockData/TrainingData/', 'NormtrainingX_' + stock_name), Npdata)
    print(num ," trainX  ", Npdata.shape)
    print(Npdata)

    Npdata = scaler.transform(resha(x_test))# normalize x_test with scale of train_x
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

    Npdata = np.array(open_money)
    np.save(os.path.join('./StockData/TrainingData/', 'opentestingX_' + stock_name), Npdata)
    print(num, " opentestX  ", Npdata.shape)
    print(Npdata)

def generate_train(feature, data, name):
    train_x = []
    train_y = []
    open_money = []
    for _, span_data in data:
        if len(span_data) == 5: #Decide the way seperate the stock data
            train_x.append(span_data.loc[:].values.tolist()) #append all feature list
            mon_open = span_data['open'][0]
            fri_close = span_data['close'][4]
            open_money.append(mon_open)
            train_y.append(fri_close - mon_open)
        else:
            continue
    save_np(train_x,train_y,open_money, name)

def add_features(df):
    add_rsv(df)
    add_k(df)
    add_d(df)
    add_MA(df)
    add_MACD(df)
    
def load_csv(num):
    index_list = []
    stock_data = pd.DataFrame(pd.read_csv('./StockData/'+num+'.csv'))
    index_data = pd.DataFrame(pd.read_csv("./usa_stock_data/usa_index.csv"))
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data.drop([0], axis = 0, inplace = True)  #drop 第一天 因為stockdata 有16年跳到17年的問題
    index_data.drop(index_data.columns[0], axis = 1, inplace = True ) #drop usa_index 的 date
    stock_data = stock_data.reset_index(drop = True)
    index_list.append(stock_data)
    index_list.append(index_data)
    return index_list

def feature_filter(df, feature):
    df = df[df.columns[df.columns.isin(feature)]] #篩選出需要的feature
    return df

stock_num = stock_dic['stock_num']
feature = stock_dic['features']
span = stock_dic['span']
close_type = stock_dic['close_type']

total_data = load_csv(stock_num) #個股[0], USA index[1]
add_features(total_data[0]) #在該個股上添加特徵
df = feature_filter(total_data[0], feature) #個股留下需要的feature
df = pd.concat(total_data, axis = 1).reindex(total_data[0].index) #將美國指數concat到個股上
df = df.dropna()
print(df)
df.to_csv('/home/zxiro/MBI/train.csv')
df = df.set_index('date').resample('w')
generate_train(feature, df, stock_num)
