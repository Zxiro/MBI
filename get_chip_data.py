import csv
import time
import pandas as pd
import requests as req
import datetime as dt
from build_config import stock_dic
from io import StringIO

##抓前三天的籌碼資料給系統
##在build_train_data裡對total data作處理篩選出特定個股的籌碼資料


day_before_yesterday = dt.date.today()- dt.timedelta(days = 2)
yesterday = dt.date.today() - dt.timedelta(days = 1)

stock_id = stock_dic['stock_num']
stock_data = pd.DataFrame(pd.read_csv('./stock'+stock_id+'.csv'))
stock_data['date'] = pd.to_datetime(stock_data['date'])
start_date = pd.to_datetime(stock_dic['date'])
end_date = pd.to_datetime(stock_dic['end_date'])
count = 0
date = []
weekly_start = [] #The first trading day of each week
for i in stock_data["date"]:
    if( start_date > i or end_date < i):
        stock_data.drop([count], axis = 0, inplace = True)
    count = count + 1
stock_data = stock_data.reset_index(drop = True)
weekly_stock_data = stock_data.set_index('date').resample('w')

for (_, cool) in weekly_stock_data:
    if len(cool) !=0 :
        tmp = str(cool.index[0]).replace("-", "")
        if(len(tmp)!=0):
            weekly_start.append(tmp[:8])

for i in range(len(stock_data['date'])):
    tmp = str(stock_data['date'][i]).replace("-", "")
    if(len(tmp)!=0):
       date.append(tmp[:8])

i = 0
#headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36 Edg/85.0.564.63"}
while i < len(stock_data['date']):
    break
    print(date[i])
    tse_csv = req.get('https://www.twse.com.tw/fund/T86?response=csv&date='+date[i]+'&selectType=ALLBUT0999')
    try:
        df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
    except Exception:
        print("parsererror: ", i)
        continue
    except IndexError:
        print("indexerror: ", i)
        continue
    except ValueError:
        print("valueerr: ", i)
        continue
    df['date'] = stock_data['date'][i]
    df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
    df['inv_trust'] = df['投信買賣超股數']
    df['dealer'] = df['自營商買賣超股數']
    df['foregin_inv'] = df.iloc[:, 4:5]
    df['total_inv_overview'] = df['三大法人買賣超股數']
    print(df)
    df.drop(df.columns[ :-6],inplace = True, axis = 1)
    df.to_csv('./chip/'+ date[i] + '.csv')
    i = i+1
    time.sleep(2)

i = 0
while i < len(weekly_start):
    print(weekly_start[i])
    tse_csv = req.get('https://www.twse.com.tw/fund/TWT54U?response=csv&date='+weekly_start[i]+'&selectType=ALLBUT0999')
    try:
        df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
    except Exception:
        print("parsererror: ", i)
        continue
    except IndexError:
        print("parsererror: ", i)
        continue
    except ValueError:
        print("valueerr: ", i)
        continue
    df['date'] = stock_data['date'][i]
    df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
    df['inv_trust'] = df['投信買賣超股數']
    df['dealer'] = df['自營商買賣超股數']
    df['foregin_inv'] = df.iloc[:, 4:5]
    df['total_inv_overview'] = df['三大法人買賣超股數']
    print(df)
    df.drop(df.columns[ :-6],inplace = True, axis = 1)
    df.to_csv('./weekly_chip/'+ weekly_start[i] + '.csv')
    i = i+1
    time.sleep(2)
