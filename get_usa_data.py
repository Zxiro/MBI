import json
import datetime as dt
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web
from build_config import index_dic
from build_config import stock_dic

def get_usa_index():
    yf.pdr_override()
    usa_dict = index_dic['index']
    features = index_dic['features']
    start = stock_dic['date']
    end = stock_dic['end_date']
    index_list = []
    for key in usa_dict:
        df = pd.DataFrame(web.get_data_yahoo(key, start, end))
        df = df[df.columns[df.columns.isin(features)]]
        df.rename(columns = { "Open":usa_dict[key]+"_open", "High":usa_dict[key]+"_high", "Low":usa_dict[key]+"_low", "Close":usa_dict[key]+"_close", "Adj Close":usa_dict[key]+"_adj_close", "Volume":usa_dict[key]+"_volume"}, inplace = True)
        index_list.append(df)
    df.reset_index(inplace = True)
    df.drop('Date', axis = 1, inplace = True)
    print(df)
    return df
