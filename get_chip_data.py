import csv
import requests
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

for i in stock_data["date"]:
    if( start_date > i or end_date < i):
        stock_data.drop([count], axis = 0, inplace = True)
    count = count + 1
stock_data = stock_data.reset_index(drop = True)

for i in range(len(stock_data['date'])):
    week_start = str(stock_data['date'][i]).replace("-", "")
    if(len(week_start)!=0):
       stock_data['date'][i] = week_start[:8]

with open("./chip.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['stock_id', 'inv_trust', 'dealer', 'foregin_inv', 'total_inv_overview'])
    for i in range(len(stock_data['date'])):
        print(stock_data['date'][i])
        tse_csv = requests.get('https://www.twse.com.tw/fund/T86?response=csv&date='+stock_data['date'][i]+'&selectType=ALLBUT0999')
        df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
        print(id(df))
        df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
        df['inv_trust'] = df['投信買賣超股數']
        df['dealer'] = df['自營商買賣超股數']
        df['foregin_inv'] = df.iloc[:, 4:5]
        df['total_inv_overview'] = df['三大法人買賣超股數']
        print(df.shape)
        #print(df)
        df.drop(df.columns[ :-5],inplace = True, axis = 1)
        mask = (df['stock_id'] == stock_id)
        print(df[mask])
        writer.writerow(df[mask].iloc[0])
        #print(df[mask].iloc[0])
        time.sleep(5.6)

