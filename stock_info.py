
import datetime
from datetime import timedelta
import json
import os
import pandas as pd
from pandas.io.json import json_normalize
import time
import numpy as np
import sys
#stock_symbol = input("stock input:")
def data_read(dirPath):  #瀏覽所有檔案
    result = [f for f in sorted(os.listdir(dirPath)) if os.path.isfile(os.path.join(dirPath, f))]
    return result
#讀取指定股票的資料
def stock_read(stock_symbol,dirPath):
    result = data_read(dirPath) #讀取資料
    if(len(stock_symbol)!=4):
        return
    first = 0;
    for i in result:
        data_locate = os.path.join(dirPath,i)
        #data = pd.read_json(data_locate)
        with open(data_locate, 'r') as f:
            data = json.load(f)
        data = data.get(stock_symbol)
        if(data == None):
            continue
        data = pd.DataFrame.from_dict(data,orient='index').T
        #data = data.T
        #data = data.loc[[stock_symbol],['open','high','low','close','volume']]
        column = data.columns.tolist()
        column.insert(0,"date")
        data = data.reindex(columns = column)
        data["date"] = i.split('.')[0]
        print(data)
        if first == 0:
            stock_data = data
            first = first + 1
        else:
            stock_data = pd.concat([stock_data,data],axis=0)
        print(stock_data)


dirPath = r"/home/db/stock_resource_center/resource/twse/json"
#dirPath = r"./stock_data"
result = [f for f in sorted(os.listdir(dirPath)) if os.path.isfile(os.path.join(dirPath, f))]
with open(os.path.join(dirPath,result[-1]),'r') as f:
    data = json.load(f)
'''stock_symbol = data.keys()
stock_symbol = list(stock_symbol)
stock_symbol.remove('id')'''
if len(sys.argv) < 2:
    stock_symbol = input('輸入股票號碼:')
else:
    stock_symbol = sys.argv[1]
stock_read(stock_symbol,dirPath)
'''for ss in stock_symbol:
    stock_read()
    file_name = "./StockData/stock"+ss+".csv"
    stock_data.to_csv(file_name,index=False)
    stock_data.drop(stock_data.index, inplace=True)
'''
