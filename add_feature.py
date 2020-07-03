import pandas as pd
from talib import abstract

class Add_feature:
    def __init__(self, df):
        self.data = pd.DataFrame(df)
        self.add_rsv()
        self.add_k()
        self.add_d()
        self.add_MA()
        self.add_MACD()

    def add_MA(self): #MA 平均價
        for ma in ['5','10', '20', '30', '60']:
            self.data["MA"+ ma] = self.data.close.rolling(int(ma)).mean()

    def add_MACD(self):
        tmp_df= abstract.MACD(self.data)
        self.data['MACD'] = tmp_df[tmp_df.columns[0]].values.tolist()
        self.data['MACDsignal'] = tmp_df[tmp_df.columns[1]].values.tolist()
        self.data['MACDhist'] = tmp_df[tmp_df.columns[2]].values.tolist()
    
    def add_rsv(self): # rsv (今天收盤-最近9天的最低價)/(最近9天的最高價-最近9天的最低價)
        rsv=[]
        close=self.data.loc[ : ,'close'].values.tolist() 
        for i in range(0,len(close)):
            if i>=8:
                low=min(self.data.loc[ i-8 : i ,'low'].values.tolist())
                high=max(self.data.loc[ i-8 : i ,'high'].values.tolist()) 
                rsv.append(((close[i]-low)/(high-low))*100)
            else:
                rsv.append(0)
        self.data['rsv'] = rsv

    def add_k(self):  # k (2/3昨日K 加 1/3 今日rsv)
        k=[] 
        rsv=self.data.loc[ : ,'rsv'].values.tolist() 
        for i in range(0,len(rsv)):
            if i>=1:
                k.append(((2/3)*k[i-1])+((1/3)*rsv[i]))
            else:
                k.append(rsv[0])       
        self.data['k'] = k

    def add_d(self): # d (2/3昨日d 加 1/3 今日k)
        d=[]
        k=self.data.loc[ : ,'k'].values.tolist() 
        for i in range(0,len(k)):
            if i>=1:
                d.append(((2/3)*d[i-1])+((1/3)*k[i]))
            else:
                d.append(k[0])            
        self.data['d'] = d