import json

dic = {
    'stock_num':'stock0050',
    'date':'2004-02-12', #永豐給lab最早的資料時間
    'features' :  ['close', 'open', 'high', 'low', 'volume', 'k','d','rsv','MA5','MA30','MA60'],
    'usa_index':['nas', 'sox', 'dji'],  #dji,nas,sox,sp
    'span':{'d':1, 'w':5, 'm': 30}, # 'span' : ['d','w','m'],
    'close_type' :'close',#'close_type' : ['close','adj_close']
    'usa_close_type':'Adj Close'
}

file="config.json"
with open(file,'w') as fileobj:
    json.dump(dic,fileobj)

print("Finish change")