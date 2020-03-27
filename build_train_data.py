import datetime #convert data to datetime64(ns)
import glob
import talib
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import mplfinance as mpf
from statistics import mean
from sklearn import preprocessing

def add_MA(data):
    for ma in ['5', '20', '30', '60']:
        data["MA"+ ma] = data.close.rolling(int(ma)).mean()

def add_rsv(data): # rsv (今天收盤-最近9天的最低價)/(最近9天的最高價-最近9天的最低價)
    rsv=[]
    close=data.loc[ : ,'close'].values.tolist() 
    for i in range(0,len(close)):
        if i>=8:
            low=min(data.loc[ i-8 : i ,'low'].values.tolist())
            high=max(data.loc[ i-8 : i ,'high'].values.tolist()) 
            rsv.append(((close[i]-low)/(high-low))*100)
        else:
            rsv.append(0) 
    return rsv

def add_k(data):  # k (2/3昨日K 加 1/3 今日rsv)
    k=[] 
    rsv=data.loc[ : ,'rsv'].values.tolist() 
    for i in range(0,len(rsv)):
        if i>=1:
            k.append(((2/3)*k[i-1])+((1/3)*rsv[i]))
        else:
            k.append(rsv[0])       
    return k

def add_d(data): # d (2/3昨日d 加 1/3 今日k)
    d=[]
    k=data.loc[ : ,'k'].values.tolist() 
    for i in range(0,len(k)):
        if i>=1:
            d.append(((2/3)*d[i-1])+((1/3)*k[i]))
        else:
            d.append(k[0])            
    return d

def add_USA_index(data):
    return 0

def resha(x): #從 (幾周,每周幾天,特徵數)reshape成(天*周,特徵數) ->(總天數,特徵數)
    nptrain = np.array(x)
    nptrain = np.reshape(nptrain,(nptrain.shape[0]*nptrain.shape[1], nptrain.shape[2]))
    return nptrain

def save_np(x,y,open_money):
    path = './StockData/stock0050.csv'
    #train_x, x_test,train_y, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
    train_x = x[:-50]
    train_y = y[:-50]
    x_test = x[-50:]
    y_test = y[-50:]
    open_money = open_money[-50:]
    stock_name = path[12:21]
    scaler = preprocessing.StandardScaler() #初始化scaler
    scaler = scaler.fit(resha(train_x))  #  標準化後的數據
    train_x = scaler.transform(resha(train_x))
    Npdata = train_x
    np.save(os.path.join('./StockData/TrainingData/', 'NormtrainingX_' + stock_name), Npdata)
    print(path[12:21] ," trainX  ", Npdata.shape)
    print(Npdata)
    Npdata = scaler.transform(resha(x_test))# normalize x_test with scale of train_x
    np.save(os.path.join('./StockData/TrainingData/', 'NormtestingX_' + stock_name), Npdata)
    print(path[12:21], " testX  ", Npdata.shape) 
    print(Npdata)
    Npdata = np.array(train_y)
    np.save(os.path.join('./StockData/TrainingData/', 'trainingY_' + stock_name), Npdata)
    print(path[12:21], " trainY  ", Npdata.shape) 
    print(Npdata)
    Npdata = np.array(y_test)
    np.save(os.path.join('./StockData/TrainingData/', 'testingY_' + stock_name), Npdata)
    print(path[12:21], " testY  ", Npdata.shape) 
    print(Npdata)
    Npdata = np.array(open_money)
    np.save(os.path.join('./StockData/TrainingData/', 'opentestingX_' + stock_name), Npdata)
    print(path[12:21], " opentestX  ", Npdata.shape) 
    print(Npdata)

def generate_train(close_type, feature, data):
    train_x = []
    train_y = []
    open_money = []
    for _, span_data in data:
        span_data = span_data.dropna()
        if len(span_data) == 5: #Decide the way seperate the stock data
            xlist = span_data.loc[:, feature].values.tolist()
            train_x.append(xlist) 
            mon_open = span_data['open'][0]
            fri_close = span_data['close'][4]
            open_money.append(((mon_open)))
            train_y.append((( fri_close - mon_open)))
        else:
            continue
    #print(train_x)
    #print(train_y)

    save_np(train_x,train_y,open_money)

def add_features(csv_data):
    csv_data['rsv'] = add_rsv(csv_data)
    csv_data['k'] = add_k(csv_data)
    csv_data['d'] = add_d(csv_data)
    add_MA(csv_data)
    csv_data = csv_data.set_index('date').resample('w')
    return csv_data
    

def load_csv():
    stock_data = pd.DataFrame(pd.read_csv('./StockData/stock0050.csv'))
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    index_list = glob.glob(r"./usa_stock_data/*.csv")
    for i in range(0,4):
        index_list[i] = pd.DataFrame(pd.read_csv(index_list[i]))
        index_list[i] = index_list[i].drop(index_list[i].columns[0], axis=1)
    stock_data = stock_data.drop([0], axis=0) #drop 第一天 因為stockdata 有16年跳到17年的問題
    stock_data = stock_data.reset_index(drop=True)
    index_list.insert(0,stock_data)
    return stock_data, index_list

with open("./config.json",'r') as load_f:
    config = json.load(load_f)

feature = config['features']
print("features: " , feature)
span = config['span']
print("span: " , span)
close_type = config['close_type']
print("close_type: " , close_type)

df, USA = load_csv() #df 股票資料  USA 股票加上USA index
re = pd.concat(USA, axis=1).reindex(USA[0].index)
print(re)
re.to_csv('/home/zxiro/MBI/feature/re.csv')
exit()
df = add_features(df)
generate_train(close_type,feature,df)