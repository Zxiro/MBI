import json

stock_dic = {
    'stock_num':'0050',
    'date':'2004-02-12', #永豐給lab最早的資料時間
    'end_date':'2020-02-11',
    'features' :  ['date', 'close', 'open', 'high', 'low', 'volume', 'k','d','rsv','MA5','MA30','MA60', 'MACD', 'MACDsignal', 'MACDhist'],
    'span':['d', 'w', 'm'],
    'close_type' :'close', #['close','adj_close']
}

index_dic = {
    'index': {'^SOX':'sox'},#'^SOX':'sox', '^IXIC':'nas''^DJI':'dji' 'GSPC':'sp',
    'features':['Close', 'Open', 'High', 'Low', 'Volume'],
}
