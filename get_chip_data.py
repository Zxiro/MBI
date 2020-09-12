import requests
import pandas as pd
from build_config import stock_dic
from io import StringIO

tse_csv = requests.get('https://www.twse.com.tw/fund/TWT54U?response=csv&date=20200907&selectType=ALLBUT0999')
print('fetch')
df = pd.read_csv(StringIO(tse_csv.text), header = 1).dropna(how='all', axis=1).dropna(how='any')
df['stock_id'] = df['證券代號'].str.replace('=','').str.replace('"','')
df = df.drop(['證券代號'], axis=1)
mask = (df['stock_id'] =="0050")
print(df[mask])

