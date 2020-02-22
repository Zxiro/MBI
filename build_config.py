import json

dic = {
    'features' :  ['k','d','rsv','ma5','ma20','ma60'],
    'span':'w', # 'span' : ['d','w','m'],
    'close_type' :'close'#'close_type' : ['close','adj_close']
}

file="config.json"
with open(file,'w') as fileobj:
    json.dump(dic,fileobj)

print("Config changed")