import requests
import pandas as pd
from build_config import stock_dic
from io import StringIO

stock_id = stock_dic['stock_num']
tse_csv = requests.get('https://www.twse.com.tw/fund/TWT54U?response=csv&date=20200907&selectType=ALLBUT0999')

df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
df['inv_trust'] = df['投信買賣超股數']
df['dealer'] = df['自營商買賣超股數']
df['institutional_inv_overview'] = df['三大法人買賣超股數']
df = df.drop(['投信買賣超股數'], axis=1)
df = df.drop(['證券代號'], axis=1)
df = df.drop(['自營商買賣超股數'], axis=1)
df = df.drop(['證券名稱'], axis=1)
df = df.drop(['三大法人買賣超股數'], axis=1)
mask = (df['stock_id'] == stock_id)
print(df[mask])

