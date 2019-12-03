import datetime
from datetime import timedelta
import json
import os
import pandas as pd
from pandas.io.json import json_normalize
import time
import numpy as np

stock_symbol = input("stock input:")

dirPath = r"/home/db/stock_resource_center/resource/twse/json"
#dirPath = r"./stock_data"
result = [f for f in sorted(os.listdir(dirPath)) if os.path.isfile(os.path.join(dirPath, f))]

for i in result:
    data_locate = os.path.join(dirPath,i)
    data = pd.read_json(data_locate)
    data = data.T
    data = data.loc[[stock_symbol],['open','high','low','close','volume']]
    column = data.columns.tolist()
    column.insert(0,"date")
    data = data.reindex(columns = column)
    data["date"] = i.split('.')[0]
    print(data)
    if i == result[0]:
        stock_data = data
    else:
        stock_data = pd.concat([stock_data,data],axis=0)
'''
for i in last_date:
    data_locate = "./stock_data/"+i+".json"
    data = pd.read_json(data_locate)
    data = data.T
    data = data.loc[[stock_symbol],['high','low','open','close']]
    column = data.columns.tolist()
    column.insert(0,"date")
    data = data.reindex(columns = column)
    data["date"] = i;
    print(data)
    if i == last_date[0]:
        stock_data = data
    else:
        stock_data = pd.concat([stock_data,data],axis=0)
'''
'''
date = "2019-09-02"
data_locate = "./stock_data/"+date+".json"
data = pd.read_json(data_locate)
data = data.T
print(data.loc[[stock_symbol],['high','low','open','close']])
stock_data = data.loc[[stock_symbol],['high','low','open','close']]
'''
print(stock_data)
file_name = "stock_"+stock_symbol+".csv"
stock_data.to_csv(file_name,index=False)
