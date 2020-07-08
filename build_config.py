import json

stock_dic = {
    'stock_num':'0050',
    'date':'2009-02-11', #永豐給lab最早的資料時間
    'end_date':'2019-02-11',
    'features' :  ['date', 'close', 'open', 'high', 'low', 'volume', 'k','d','rsv','MA5','MA30','MA60', 'MACD', 'MACDsignal', 'MACDhist'],
    'span':['d', 'w', 'm'],
    'close_type' :'close', #['close','adj_close']
}

index_dic = {
    'index': {'^GSPC':'sp'},#'^SOX':'sox', '^IXIC':'nas''^DJI':'dji',
    'features':['Close', 'Open', 'High', 'Low', 'Volume'],
}
