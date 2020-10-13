import csv
import requests as req
import time
import pandas as pd
from build_config import stock_dic
from io import StringIO

stock_id = stock_dic['stock_num']
stock_data = pd.DataFrame(pd.read_csv('./StockData/stock'+stock_id+'.csv'))
stock_data['date'] = pd.to_datetime(stock_data['date'])
start_date = pd.to_datetime(stock_dic['date'])
end_date = pd.to_datetime(stock_dic['end_date'])
count = 0
date = []
for i in stock_data["date"]:
    if( start_date > i or end_date < i):
        stock_data.drop([count], axis = 0, inplace = True)
    count = count + 1
stock_data = stock_data.reset_index(drop = True)

for i in range(len(stock_data['date'])):
    week_start = str(stock_data['date'][i]).replace("-", "")
    if(len(week_start)!=0):
       date.append(week_start[:8])
'''with open("./Chip_data/"+ stock_id +"daily_chip.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['stock_id', 'inv_trust', 'dealer', 'foregin_inv', 'total_inv_overview'])
    i = 0'''
i = 0
headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36 Edg/85.0.564.63"}
while i < len(stock_data['date']):
    print(date[i])
    tse_csv = req.get('https://www.twse.com.tw/fund/T86?response=csv&date='+date[i]+'&selectType=ALLBUT0999', headers = headers)
    try:
        df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
    #print(id(df))
    except Exception:
        print("parsererror: ", i)
        continue
    except IndexError:
        print("parsererror: ", i)
        continue
    df['date'] = stock_data['date'][i]
    df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
    df['inv_trust'] = df['投信買賣超股數']
    df['dealer'] = df['自營商買賣超股數']
    df['foregin_inv'] = df.iloc[:, 4:5]
    df['total_inv_overview'] = df['三大法人買賣超股數']
    print(df.shape)
    #print(df)
    df.drop(df.columns[ :-6],inplace = True, axis = 1)
    #mask = (df['stock_id'] == stock_id)
    #print(df[mask])
    #writer.writerow(df[mask].iloc[0])
    #print(df[mask].iloc[0])i
    df.to_csv('./Chip_data/daily_chip_data')
    i = i+1
    time.sleep(2)
'''with open("./chip_data/"+stock_id+"week_chip.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['stock_id', 'inv_trust', 'dealer', 'foregin_inv', 'total_inv_overview'])
    i = 0
    while i < len(stock_data['date']):
        print(date[i])
        tse_csv = req.get('https://www.twse.com.tw/fund/TWT54U?response=csv&date='+date[i]+'&selectType=ALLBUT0999')
        try:
            df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
        #print(id(df))
        except Exception:
            print("parsererror: ", i)
            continue
        except IndexError:
            print("parsererror: ", i)
            continue
        df['date'] = stock_data['date'][i]
        df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
        df['inv_trust'] = df['投信買賣超股數']
        df['dealer'] = df['自營商買賣超股數']
        df['foregin_inv'] = df.iloc[:, 4:5]
        df['total_inv_overview'] = df['三大法人買賣超股數']
        print(df.shape)
        df.drop(df.columns[ :-6],inplace = True, axis = 1)
        mask = (df['stock_id'] == stock_id)
        print(df[mask])
        writer.writerow(df[mask].iloc[0])
        i = i+1
        time.sleep(2)'''
