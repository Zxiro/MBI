import datetime#import datetime module convert data to datetime64(ns)
import os
import glob
import json
import numpy as np
import pandas as pd#import pandas module(Better than csv module!)
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import mean

def add_MA5(data):
    close=data.loc[ : ,'close'].values.tolist()
    ma5=[]
    for i in range(0,len(close)):
        if i >= 4:
            ma5.append(sum(data.loc[i-4:i, 'close'].values.tolist())/5)
        else:
            ma5.append(0)
    return ma5

def add_MA20(data):
    close=data.loc[:, 'close'].values.tolist()
    ma20=[]
    for i in range(0,len(close)):
        if i>=20:
            ma20.append(sum(data.loc[i-20:i,'close'].values.tolist())/20)
        else:
            ma20.append(0)
    return ma20

def add_MA60(data):
    close=data.loc[ : ,'close'].values.tolist()
    ma60=[]
    for i in range(0,len(close)):
        if i>=60:
            ma60.append(sum(data.loc[ i-60 : i ,'close'].values.tolist())/60)
        else:
            ma60.append(0)
    return ma60

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

def resha(x): #從 (幾周,每周幾天,特徵數)reshape成(天*周,特徵數) ->(總天數,特徵數)
    nptrain = np.array(x)
    nptrain = np.reshape(nptrain,(nptrain.shape[0]*nptrain.shape[1], nptrain.shape[2]))
    return nptrain
 
def save_np(x,y):
    path = './StockData/stock0056.csv'
    train_x, x_test,train_y, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
    stock_name = path[12:21] 
    scaler = sk.preprocessing.StandardScaler() 
    scaler = scaler.fit(resha(train_x))  #  標準化後的數據 
    train_x = scaler.transform(resha(train_x))
    Npdata = train_x
    np.save(os.path.join('./StockData/TrainingData/','NormtrainingX_'+stock_name),Npdata)
    print(path[12:21]," trainX  ",Npdata.shape)#print(Npdata)
    Npdata = scaler.transform(resha(x_test))# normalize x_test with scale of train_x
    np.save(os.path.join('./StockData/TrainingData/','NormtestingX_'+stock_name),Npdata)
    print(path[12:21]," testX  ",Npdata.shape) #print(Npdata)
    Npdata = np.array(train_y)
    np.save(os.path.join('./StockData/TrainingData/','trainingY_'+stock_name),Npdata)
    print(path[12:21]," trainY  ",Npdata.shape) #print(Npdata)    
    Npdata = np.array(y_test)
    np.save(os.path.join('./StockData/TrainingData/','testingY_'+stock_name),Npdata)
    print(path[12:21]," testY  ",Npdata.shape) #print(Npdata)
    
def generate_train(close_type,feature,data):
    train_x = []
    train_y = []   
    for _, date in data:
        date = date.dropna()
        if len(date)==5: #Decide the way seperate the stock data
            xlist = date.loc[ : , feature].values.tolist()
            for i  in range(0,len(xlist)):
                for k  in range(0,len(xlist[i])):
                    xlist[i][k] = "%.2f" %(xlist[i][k])
            train_x.append(xlist) 
            mon_open = date['open'][0]
            fri_close = date['close'][4]
            train_y.append("%.2f" %(( fri_close - mon_open)))
        else:
            continue
    #print(train_x)
    #print(train_y)
    save_np(train_x,train_y)

def add_features(csv_data):
    csv_data['rsv'] = add_rsv(csv_data)
    csv_data['k'] = add_k(csv_data)
    csv_data['d'] = add_d(csv_data)
    csv_data['ma5'] = add_MA5(csv_data)
    csv_data['ma20'] = add_MA20(csv_data)
    csv_data['ma60'] = add_MA60(csv_data)
    csv_data = csv_data.set_index('date').resample('w')#print(type(csv_data))
    return csv_data
    

def load_csv():
    #for path in glob.glob(r'./StockData/stock0050.csv'):
    csv_data = pd.DataFrame(pd.read_csv('./StockData/stock0050.csv'))
    csv_data['date'] = pd.to_datetime(csv_data['date'])
    csv_data = csv_data.drop([0],axis=0)#drop 第一天 因為stockdata 有16年跳到17年的問題
    csv_data = csv_data.reset_index(drop=True)
    return csv_data


with open("./config.json",'r') as load_f:
    config = json.load(load_f)

feature = config['features']
print("features: " , feature)
span = config['span']
print("span: " , span)
close_type = config['close_type']
print("close_type: " , close_type)

df = load_csv()
df = add_features(df)

generate_train(close_type,feature,df)