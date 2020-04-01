import json
import datetime as dt
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

def adjust_df(df):
    df.reset_index(drop=True, inplace=True)
    df.drop(close_type, axis=1, inplace=True)
    return df

yf.pdr_override()
usa_dict = {'^DJI':'dji','^GSPC':'sp', '^IXIC':'nas', '^SOX':'sox'}
with open("./config.json",'r') as load_f:
    config = json.load(load_f)
close_type = config['usa_close_type']
start = config['date']
end = dt.datetime.now()

for key in usa_dict:
    df = pd.DataFrame(web.get_data_yahoo(key, start, end))
    df = adjust_df(df)
    df.rename(columns = { "Open":usa_dict[key]+"_open", "High":usa_dict[key]+"_high", "Low":usa_dict[key]+"_low", "Close":usa_dict[key]+"_close", "Adj Close":usa_dict[key]+"_adj_close", "Volume":usa_dict[key]+"_volume"}, inplace = True)
    print(df)
    df.to_csv('/home/zxiro/MBI/usa_stock_data/'+usa_dict[key]+'.csv')

print("Fetch completed")