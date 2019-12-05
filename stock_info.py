%load StockInfo.ipynb
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

for i in result
    data_locate = os.path.join(dirPath,i)
    #data = pd.read_json(data_locate)
    with open(data_locate, 'r') as f:
        data = json.load(f)
    data = data.get(stock_symbol)
    data = pd.DataFrame.from_dict(data,orient='index').T
    #data = data.T
    #data = data.loc[[stock_symbol],['open','high','low','close','volume']]
    column = data.columns.tolist()
    column.insert(0,"date")
    data = data.reindex(columns = column)
    data["date"] = i.split('.')[0]
    print(data)
    if i == result[0]:
        stock_data = data
    else:
        stock_data = pd.concat([stock_data,data],axis=0)

print(stock_data)
file_name = "stock"+stock_symbol+".csv"
stock_data.to_csv(file_name,index=False)
