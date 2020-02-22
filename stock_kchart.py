import datetime#import datetime module convert data to datetime64(ns)
import time
import os
import numpy as np
import pandas as pd#import pandas module(Better than csv module!)
import matplotlib.pyplot as plt
import mpl_finance as mpf
from twstock import Stock
from PIL import Image

localtime=time.localtime(time.time())
mon_span=input("取幾個月前的? ")
if localtime.tm_mon<int(mon_span):
    yrs=localtime.tm_year-1
    mon=12-(int(mon_span)-localtime.tm_mon)
else:
    yrs=localtime.tm_year
    mon=localtime.tm_mon

print(yrs,mon)

stock_num=input("Enter Stock: ")

for i in range(1,6):
    time.sleep(1)
    print("Wait.... ",5-i,"second")

stock=Stock(stock_num)
print("First Stage complete ")

data=pd.DataFrame(stock.fetch_from(yrs,mon))
data.set_index('date',inplace=True)
data.index=data.index.format(formatter=lambda x: x.strftime('%Y-%m-%d'))
print("Fetching complete")

fig = plt.figure(figsize=(24,16))
#用add_axes創建副圖框
ax = fig.add_axes([0.05,0.25,0.95,0.75]) 
ax2 = fig.add_axes([0.05,0.05,0.95,0.25]) 
ax.set_title(stock_num+"KChart",fontsize=12)
ax2.set_xticks(range(0, len(data.index), 10))
ax2.set_xticklabels(data.index[::10],rotation=0)
#使用mpl_finance套件
mpf.candlestick2_ochl(ax, data['open'], data['close'], data['high'],data['low'], width=0.5, colorup='r', colordown='g', alpha=1)
mpf.volume_overlay(ax2,  data['open'], data['close'], data['capacity'], width=0.5, colorup='r', colordown='g', alpha=1)
plt.savefig("C:/Users/User/Desktop/Kline/" + stock_num +".png")