import json
import datetime as dt
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

yf.pdr_override()

def adjust_df(df):
    df.reset_index(drop=True, inplace=True)
    df.drop(close_type, axis=1, inplace=True)
    return df

with open("./config.json",'r') as load_f:
    config = json.load(load_f)
close_type = config['usa_close_type']
start = config['date']
end = dt.datetime.now()
dji = pd.DataFrame(web.get_data_yahoo('^DJI', start, end))
dji = adjust_df(dji)
dji.rename(columns = { "Open":"dji_open", "High":"dji_high", "Low":"dji_low", "Close":"dji_close", "Adj Close":"dji_adj_close", "Volume":"dji_volume"}, inplace = True)
print(dji)
dji.to_csv('/home/zxiro/MBI/usa_stock_data/DJI.csv')
sp500 = pd.DataFrame(web.get_data_yahoo(['^GSPC'],start, end))
sp500 = adjust_df(sp500)
sp500.rename(columns = {"Date": "date","Open":"sp_open", "High":"sp_high", "Low":"sp_low", "Close":"sp_close", "Adj Close":"sp_adj_close", "Volume":"sp_volume"}, inplace = True)
print(sp500)
sp500.to_csv('/home/zxiro/MBI/usa_stock_data/SP500.csv')
NASDAQ = pd.DataFrame(web.get_data_yahoo(['^IXIC'],start, end))
NASDAQ = adjust_df(NASDAQ)
NASDAQ.rename(columns = {"Date": "date","Open":"nas_open", "High":"nas_high", "Low":"nas_low", "Close":"nas_close", "Adj Close":"nas_adj_close", "Volume":"nas_volume"}, inplace = True)
print(NASDAQ)
NASDAQ.to_csv('/home/zxiro/MBI/usa_stock_data/NASDAQ.csv')
SOX = pd.DataFrame(web.get_data_yahoo(['^SOX'],start, end))
SOX = adjust_df(SOX)
SOX.rename(columns = {"Date": "date","Open":"sox_open", "High":"sox_high", "Low":"sox_low", "Close":"sox_close", "Adj Close":"sox_adj_close", "Volume":"sox_volume"}, inplace = True)
print(SOX)
SOX.to_csv('/home/zxiro/MBI/usa_stock_data/SOX.csv')