import json

dic = {
    'date':'2004-02-11',#永豐給lab最早的資料時間
    'features' :  ['close', 'open', 'high', 'low', 'volume', 'k','d','rsv','MA5','MA20','MA60'],
    'span':'d', # 'span' : ['d','w','m'],
    'close_type' :'close'#'close_type' : ['close','adj_close']
}

file="config.json"
with open(file,'w') as fileobj:
    json.dump(dic,fileobj)

print("Finish change")