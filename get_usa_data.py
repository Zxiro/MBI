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
dji.rename(columns = { "Open":"dji_open", "High":"dji_high", "Low":"dji_low", "Close":"dji_close", "Adj Close":"dji_adj_close"}, inplace = True)
dji.reset_index(drop=True, inplace=True)
dji.drop("Volume", axis = 1, inplace=True)
print(dji)
dji.to_csv('/home/zxiro/MBI/usa_stock_data/DJI.csv')
sp500 = pd.DataFrame(web.get_data_yahoo(['^GSPC'],start, end))
sp500.rename(columns = {"Date": "date","Open":"sp_open", "High":"sp_high", "Low":"sp_low", "Close":"sp_close", "Adj Close":"sp_adj_close"}, inplace = True)
sp500.drop("Volume", axis = 1, inplace=True)
sp500.reset_index(drop=True, inplace=True)
print(sp500)
sp500.to_csv('/home/zxiro/MBI/usa_stock_data/SP500.csv')
NASDAQ = pd.DataFrame(web.get_data_yahoo(['^IXIC'],start, end))
NASDAQ.rename(columns = {"Date": "date","Open":"nas_open", "High":"nas_high", "Low":"nas_low", "Close":"nas_close", "Adj Close":"nas_adj_close"}, inplace = True)
NASDAQ.drop("Volume", axis = 1, inplace=True)
NASDAQ.reset_index(drop=True, inplace=True)
print(NASDAQ)
NASDAQ.to_csv('/home/zxiro/MBI/usa_stock_data/NASDAQ.csv')
SOX = pd.DataFrame(web.get_data_yahoo(['^SOX'],start, end))
SOX.rename(columns = {"Date": "date","Open":"sox_open", "High":"sox_high", "Low":"sox_low", "Close":"sox_close", "Adj Close":"sox_adj_close"}, inplace = True)
SOX.drop("Volume", axis = 1, inplace=True)
SOX.reset_index(drop=True, inplace=True)
print(SOX)
SOX.to_csv('/home/zxiro/MBI/usa_stock_data/SOX.csv')