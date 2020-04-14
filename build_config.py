import json

stock_dic = {
    'stock_num':'stock0050',
    'date':'2004-02-12', #永豐給lab最早的資料時間
    'features' :  ['date', 'close', 'open', 'high', 'low', 'volume', 'k','d','rsv','MA5','MA30','MA60', 'MACD', 'MACDsignal', 'MACDhist'],
    'span':['d', 'w', 'm'],
    'close_type' :'close',#['close','adj_close']
}

index_dic = {
    'index': {'^GSPC':'sp'},#'^SOX':'sox', '^IXIC':'nas''^DJI':'dji',
    'features':['Close', 'Open', 'High', 'Low', 'Volume'],
}

file="config.json"
with open(file,'w') as fileobj:
    json.dump(stock_dic,fileobj)

print("Finish change")