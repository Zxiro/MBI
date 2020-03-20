import json
import datetime as dt
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

yf.pdr_override()

with open("./config.json",'r') as load_f:
    config = json.load(load_f)

start = config['date']
end = dt.datetime.now()
dji = pd.DataFrame(web.get_data_yahoo('^DJI', start, end))
dji.drop("Volume", axis = 1, inplace=True) #要加上inplace = true 才能drop
print(dji)
dji.to_csv('/home/zxiro/MBI/usa_stock_data/DJI.csv')
sp500 = pd.DataFrame(web.get_data_yahoo(['^GSPC'],start, end))
sp500.drop("Volume", axis = 1, inplace=True)
print(sp500)
sp500.to_csv('/home/zxiro/MBI/usa_stock_data/SP500.csv')
NASDAQ = pd.DataFrame(web.get_data_yahoo(['^IXIC'],start, end))
NASDAQ.drop("Volume", axis = 1, inplace=True)
print(NASDAQ)
NASDAQ.to_csv('/home/zxiro/MBI/usa_stock_data/NASDAQ.csv')
SOX = pd.DataFrame(web.get_data_yahoo(['^SOX'],start, end))
SOX.drop("Volume", axis = 1, inplace=True)
print(SOX)
SOX.to_csv('/home/zxiro/MBI/usa_stock_data/SOX.csv')