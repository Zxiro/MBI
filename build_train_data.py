import csv
import datetime
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

def seperate_tr_te(lis):
    te = lis[-90:]
    tr = lis[:-90]
    return tr, te

def save_np(x, y, num, span, open_price, close_price):
    scale = preprocessing.StandardScaler()
    train_x, x_test = seperate_tr_te(x)
    train_y, y_test = seperate_tr_te(y)
    open_price = open_price[-90:]
    close_price = close_price[-90:]
    stock_name = num

    scale = scale.fit(resha(train_x))  #  standard scale, resha(x) = two dim /  (tr + te)
    with open('./csv_data/mean_var.csv', 'w', newline='')as csvfile:
        writer = csv.writer(csvfile, delimiter = ' ')
        writer.writerow(['mean', 'deviation'])
        writer.writerow([stock_dic['date'], stock_dic['end_date'], scale.mean_, scale.var_])

    x_test = scale.transform(resha(x_test)) # two dim standardlized
    train_x = scale.transform(resha(train_x))

    x_test = x_test.reshape((int(x_test.shape[0]/span), span, -1)) # return to three dim
    train_x  = train_x.reshape((int(train_x.shape[0]/span), span, -1))

    Npdata = train_x
    np.save(os.path.join('./stock_data/trx/', 'train_x_' + stock_name), Npdata)
    print(num ," train_x_: ", Npdata.shape)

    Npdata = x_test
    np.save(os.path.join('./stock_data/tex/', 'test_x_' + stock_name), Npdata)
    print(" test_x_: ", Npdata.shape)
    # print(Npdata)

    Npdata = np.array(train_y)
    np.save(os.path.join('./stock_data/try/', 'train_y_' + stock_name), Npdata)
    print( " train_y_: ", Npdata.shape)
    # print(Npdata)

    Npdata = np.array(y_test)
    np.save(os.path.join('./stock_data/tey/', 'test_y_' + stock_name), Npdata)
    print(" test_y_: ", Npdata.shape)
    #print(Npdata)

    npdata = np.array(open_price)
    np.save(os.path.join('./stock_data/trx/', 'open_x_' + stock_name), npdata)
    #print(num, " opentestx  ", npdata.shape)
    #print(npdata)'''

    npdata = np.array(close_price)
    np.save(os.path.join('./stock_data/trx/', 'close_x_' + stock_name), npdata)
    #print(num, " closetestx  ", npdata.shape)
    #print(npdata)'''

def generate_train_in_day(feature, data, name, span):
    train_x = []
    train_y = []
    open_price = []
    close_price = []
    data.drop(['date'], axis = 1, inplace = True)
    for i in range(len(data)):#everyday
        if i < len(data) and ((i + span) < len(data)):
            train_x.append([data.iloc[i+j].values.tolist() for j in range(span)])
            i += span
            Open = data.iloc[i]['open']
            Close = data.iloc[i]['close']
            open_price.append(Open)
            close_price.append(Close)
            if Close - Open >0:
                train_y.append(1)
            else:
                train_y.append(0)
            #train_y.append(Close-Open)#the next day's diff

    save_np(train_x, train_y, name, span, open_price, close_price)

def filter_feature(df, feature):
    df = df[df.columns[df.columns.isin(feature)]] #篩選出需要的feature
    print(df)
    return df

def load_csv(num, start, end):
    stock_data = pd.DataFrame(pd.read_csv('./stock_data/stock'+num+'.csv'))
    chip_data = pd.DataFrame(pd.read_csv('./chip_data/'+ num +'_daily_chip.csv'))
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    chip_data['date'] = pd.to_datetime(chip_data.index)
    chip_data.drop(['stock_id'], axis = 1, inplace = True)
    chip_data.set_index('date', inplace = True)
    chip_data = chip_data.reset_index(drop = True)

    for c in chip_data.columns:
       chip_data[c] = chip_data[c].str.replace(',','')
       chip_data[c] = chip_data[c].astype(float) #str to float
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    count = 0

    for i in stock_data["date"]:
        if( start_date > i or end_date < i):
            stock_data.drop([count], axis = 0, inplace = True)
        count = count + 1
    stock_data = stock_data.reset_index(drop = True)
    return stock_data, chip_data

if '__main__' == __name__:
    stock_num = stock_dic['stock_num']
    feature = stock_dic['features']
    span = stock_dic['span']
    close_type = stock_dic['close_type']
    start_date = stock_dic['date']
    end_date = stock_dic['end_date']
    #Basic parameter


    usa = get_usa_index() #get usa index data
    stock_data, chip_data = load_csv(stock_num, start_date, end_date) #load selected stock's data which is in the set timespan
    af = Add_feature(stock_data) #calculate the wanted feature and add on the stock dataframe
    af.data = filter_feature(af.data, feature) #leave the wanted feature
    df = pd.concat([af.data, usa], axis=1).reindex(af.data.index) #concat the USA index on the data
    df = pd.concat([af.data, chip_data], axis=1).reindex(af.data.index) #concat the chip on the data
    df = df.dropna()
    print(df)
    df.to_csv('./csv_data/train.csv')
    #df = df.set_index('date').resample('w')
    generate_train_in_day(feature, df, stock_num, span)
