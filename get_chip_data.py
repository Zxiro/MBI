import csv
import calendar
import time
import pandas as pd
import requests as req
import datetime as dt
from io import StringIO

daily_start_date = '2012-05-02' #The first day of TWSE sys to have daily chip data
daily_date = pd.bdate_range(daily_start_date, dt.date.today()).tolist()
weekly_start_date = '2002-12-30' #The first day of TWSE sys to have weekly chip data
weekly_date = pd.date_range(weekly_start_date, dt.date.today(), freq = 'W').tolist()
for i in range(len(daily_date)):
    daily_date[i] = str(daily_date[i]).replace("-", "")[:8]

for i in range(len(weekly_date)):
    weekly_date[i] = str(weekly_date[i]).replace("-", "")[:8]
i = 0
#headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36 Edg/85.0.564.63"}
while i < len(daily_date):
    print(daily_date[i])
    # tse_csv = req.get('https://www.twse.com.tw/fund/T86?response=csv&date=20120802&selectType=ALLBUT0999', timeout = 2)
    tse_csv = req.get('https://www.twse.com.tw/fund/T86?response=csv&date='+daily_date[i]+'&selectType=ALLBUT0999')
    if len(StringIO(tse_csv.text).getvalue())== 2:
        i+=1
        print("empty")
        time.sleep(2)
        continue
    try:
        df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
    except IndexError:
        print("index_error: ", i)
        time.sleep(2)
        continue
    except ValueError:
        print("value_err: ", i)
        time.sleep(2)
        continue
    except KeyError:
        print("key_err: ", i)
        time.sleep(2)
        continue
    except Exception:
        print("empty_err: ", i)
        time.sleep(2)
        continue
    # df['date'] = stock_data['date'][i]
    df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
    df['inv_trust'] = df['投信買賣超股數']
    df['dealer'] = df['自營商買賣超股數']
    df['foregin_inv'] = df.iloc[:, 4:5]
    df['total_inv_overview'] = df['三大法人買賣超股數']
    df.drop(df.columns[ :-5],inplace = True, axis = 1)
    print(df)
    df.to_csv('./chip/'+ daily_date[i] + '.csv')
    i = i+1
    time.sleep(4.5)

i = 0
while i < len(weekly_date) :
    print(weekly_date[i])
    tse_csv = req.get('https://www.twse.com.tw/fund/TWT54U?response=csv&date='+weekly_date[i]+'&selectType=ALLBUT0999', timeout = 2)
    try:
        df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
    except Exception:
        print("parser_err: ", i)
        time.sleep(2)
        continue
    except IndexError:
        print("index_err: ", i)
        time.sleep(2)
        continue
    except ValueError:
        print("value_err: ", i)
        time.sleep(2)
        continue
    except KeyError:
        print("key_err: ", i)
        time.sleep(2)
        continue
    # df['date'] = stock_data['date'][i]
    df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
    df['inv_trust'] = df['投信買賣超股數']
    df['dealer'] = df['自營商買賣超股數']
    df['foregin_inv'] = df.iloc[:, 4:5]
    df['total_inv_overview'] = df['三大法人買賣超股數']
    df.drop(df.columns[ :-5],inplace = True, axis = 1)
    print(df)
    df.to_csv('./weekly_chip/'+ weekly_date[i] + '.csv')
    i = i+1
    time.sleep(1.5)
