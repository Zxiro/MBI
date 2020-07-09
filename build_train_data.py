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
    print(stock_name)
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

def filter_feature(df, feature):
    df = df[df.columns[df.columns.isin(feature)]] #篩選出需要的feature

def load_csv(num):
    stock_data = pd.DataFrame(pd.read_csv('./StockData/'+num+'.csv'))
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.drop([0], axis=0)  #drop 第一天 因為stockdata 有16年跳到17年的問題
    stock_data = stock_data.reset_index(drop=True)
    return stock_data
if '__main__' == __name__:
stock_num ='stock' + stock_dic['stock_num']
feature = stock_dic['features']
span = stock_dic['span']
close_type = stock_dic['close_type']
start_date = stock_dic['date']
end_date = stock_dic['end_date']
usa = get_usa_index() #get usa index data
stock_data = load_csv(stock_num) #個股加上USA index 全部資料
af = Add_feature(stock_data)
filter_feature(af.data, feature) #在該個股上留下需要的feature
df = pd.concat([af.data, usa], axis=1).reindex(af.data.index) #將美國指數concat到個股上
df = df.dropna()
print(df)
df.to_csv('./train.csv')
df = df.set_index('date').resample('w')

generate_train(feature, df, stock_num)