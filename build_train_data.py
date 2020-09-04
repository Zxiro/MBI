import csv
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

def resha(x): #(week, day, features) reshape into (week*day, features) -> (total days, features)
    nptrain = np.array(x)
    print(nptrain.shape)
    nptrain = np.reshape(nptrain,(nptrain.shape[0] * nptrain.shape[1], nptrain.shape[2]))
    return nptrain

def seperate_tr_te(list):
    te = list[-50:]
    tr = list[:-50]
    return tr, te

def save_np(x, y, open_money, num, mon, fri):
    scale = preprocessing.StandardScaler()
    train_x, x_test = seperate_tr_te(x)
    train_y, y_test = seperate_tr_te(y)
    train_y_mon, y_test_mon = seperate_tr_te(mon)
    train_y_fri, y_test_fri = seperate_tr_te(fri)
    open_money = open_money[-50:]
    stock_name = num

    scale = scale.fit(resha(train_x))  #  標準化後的標準scale, resha(x) = two dim /  (tr + te) 

    with open('./mean_var.csv', 'w', newline='')as csvfile:
        writer = csv.writer(csvfile, delimiter = ' ')
        writer.writerow(['平均', '變異數'])
        writer.writerow([stock_dic['date'], stock_dic['end_date'], scale.mean_, scale.var_])

    '''print(scale.mean_) # mean of each feature
    print(scale.var_)  # deviation of each feature'''

    x_test = scale.transform(resha(x_test)) # two dim standradlized
    train_x = scale.transform(resha(train_x))

    x_test = x_test.reshape((int(x_test.shape[0]/5), 5, -1)) # return to three dim
    train_x  = train_x.reshape((int(train_x.shape[0]/5), 5, -1))

    Npdata = train_x
    np.save(os.path.join('./StockData/TrainingData/', 'NormtrainingX_' + stock_name), Npdata)
    print(num ," trainX: ", Npdata.shape)
    #print(Npdata)

    Npdata = x_test
    np.save(os.path.join('./StockData/TrainingData/', 'NormtestingX_' + stock_name), Npdata)
    print(" testX: ", Npdata.shape)
    #print(Npdata)

    Npdata = np.array(train_y)
    np.save(os.path.join('./StockData/TrainingData/', 'trainingY_' + stock_name), Npdata)
    print( " trainY: ", Npdata.shape)
    #print(Npdata)

    Npdata = np.array(y_test)
    np.save(os.path.join('./StockData/TrainingData/', 'testingY_' + stock_name), Npdata)
    print(" testY: ", Npdata.shape)
    #print(Npdata)

    Npdata = np.array(train_y_mon)
    np.save(os.path.join('./StockData/TrainingData/', 'trainingY_mon_' + stock_name), Npdata)
    print( " trainY_mon: ", Npdata.shape)
    #print(Npdata)

    Npdata = np.array(y_test_mon)
    np.save(os.path.join('./StockData/TrainingData/', 'testingY_mon_' + stock_name), Npdata)
    print(" testY_mon: ", Npdata.shape)
    #print(Npdata)

    Npdata = np.array(train_y_fri)
    np.save(os.path.join('./StockData/TrainingData/', 'trainingY_fri_' + stock_name), Npdata)
    print( " trainY_fri: ", Npdata.shape)
    #print(Npdata)

    Npdata = np.array(y_test_fri)
    np.save(os.path.join('./StockData/TrainingData/', 'testingY_fri_' + stock_name), Npdata)
    print(" testY_fri: ", Npdata.shape)
    #print(Npdata)

    Npdata = np.array(open_money)
    np.save(os.path.join('./StockData/TrainingData/', 'opentestingX_' + stock_name), Npdata)
    #print(num, " opentestX  ", Npdata.shape)
    #print(Npdata)

def generate_train(feature, data, name):
    gen_x = []
    gen_y = []
    train_x = []
    train_y = []
    train_y_mon = []
    train_y_fri = []
    open_money = []
    for(_, span_data) in data:
        if len(span_data)==5:
            tmp_data = span_data
            break
    k = 0
    for(_, span_data) in data:
        if len(span_data)==5:
            if(k==0):
                k=k+1
                continue
            tmp_data2 = span_data
            break
    k = 0
    for _, span_data in data:
        #預測未滿5天
        '''if len(span_data) == 5: #Decide the way seperate the stock data
            if(k == 0 or k == 1):
                k = k+1
                continue
<<<<<<< Updated upstream
            #new_span_data = pd.concat([tmp_data,span_data])#two week
            #threeweek_span_data = pd.concat([pd.concat([tmp_data,tmp_data2]),span_data])
            #tmp_data = tmp_data2
            #tmp_data2 = span_data
=======
            new_span_data = pd.concat([tmp_data,span_data])
            threeweek_span_data = pd.concat([pd.concat([tmp_data,tmp_data2]),span_data])
            train_x.append(span_data.loc[:].values.tolist())
            #train_x.append(threeweek_span_data.loc[:].values.tolist()) #append all feature list
            mon_open = span_data['open'][0]
            fri_close = span_data['close'][4]
            open_money.append(mon_open)
            train_y.append(fri_close - mon_open)
            #tmp_data = span_data
            tmp_data = tmp_data2
            tmp_data2 = span_data
>>>>>>> Stashed changes
            train_y_fri.append(fri_close)
            train_y_mon.append(mon_open)
    train_x.pop()#drop last
    train_y.pop(0)#drop first
======='''
        # copy span_data 第一份取x 第二份取y # 準備兩周的 tx ty
        gen_x.append(span_data)
        gen_y.append(span_data)
    gen_x.pop()
    gen_y.pop(0)
    for i in range(len(gen_x)):
        if(len(gen_x[i])==5 and len(gen_y[i]) != 0):
            train_x.append(gen_x[i].values.tolist()) #gen_x[i].to_numpy() / series and df all ok
            mon_open = gen_y[i]['open'][0]
            last_close = gen_y[i]['close'].iloc[-1] #iloc for row / i for index
            open_money.append(mon_open)
            train_y.append(last_close - mon_open)
            train_y_fri.append(last_close)
            train_y_mon.append(mon_open)

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
    count = 0
    for i in stock_data["date"]:
        if( start_date > i or end_date < i):
            stock_data.drop([count], axis = 0, inplace = True)
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
    print(usa)
    stock_data = load_csv(stock_num, start_date, end_date) #load selected stock's data which is in the set timespan 
    af = Add_feature(stock_data) #calculate the wanted feature and add on the stock dataframe
    af.data = filter_feature(af.data, feature) #leave the wanted feature
    df = pd.concat([af.data, usa], axis=1).reindex(af.data.index) #concat the USA index on the data
    df = df.dropna()
    print(df)
    print('------------------------')
    df.to_csv('./train.csv')
    df = df.set_index('date').resample('w')
    generate_train(feature, df, stock_num)
