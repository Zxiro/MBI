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

for i in range(len(stock_data['date'])):
    print(stock_data['date'][i])
    tse_csv = requests.get('https://www.twse.com.tw/fund/T86?response=csv&date='+stock_data['date'][i]+'&selectType=ALLBUT0999')
    df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
    df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
    df['inv_trust'] = df['投信買賣超股數']
    df['dealer'] = df['自營商買賣超股數']
    df['institution_inv_overview'] = df['三大法人買賣超股數']
    df['foregin_inv'] = df['institution_inv_overview'] - df['inv_trust'] - df['dealer']
    df.drop(df.columns[ :-5],inplace = True, axis = 1)
    mask = (df['stock_id'] == stock_id)
    print(df[mask])
    time.sleep(10)

