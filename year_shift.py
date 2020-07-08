import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
from evaluate import Evaluate

start_date = 0
end_date   = 0
start_year = 0
end_year = 0
number_of_data = 0

def load_csv(num):
    stock_data = pd.DataFrame(pd.read_csv('./StockData/stock'+num+'.csv'))
    print(stock_data['date'])
    print(stock_data['date'][0].split('-',1)[0])
    print(stock_data.iloc[-1]['date'].split('-',1)[0])
    global start_year,end_year,start_date,end_date,number_of_data
    start_year = stock_data['date'][0].split('-',1)[0]
    end_year = stock_data.iloc[-1]['date'].split('-',1)[0]
    start_date = stock_data['date'][0]
    end_date = stock_data.iloc[-1]['date']
    number_of_data = int(end_year) - int(start_year) - 10
    if(number_of_data <= 0):
        number_of_data = 1
    #print(number_of_data)

load_csv('0050')

finput = open("ten_year_evaluate.csv","w")

print(number_of_data)
#number_of_data =1
for i in range(number_of_data):

    print(i)
    if(i!=0):
        start_date = start_date.replace(start_date.split('-',1)[0],str(int(start_date.split('-',1)[0]) + 1))
    if(int(end_year)-int(start_year)-10<=0 or (str(int(start_date.split('-',1)[0])+10)==end_year)):
        end = end_date
    else:
        end = start_date.replace(start_date.split('-',1)[0],str(int(start_date.split('-',1)[0]) + 10))
    print(start_date)
    print(end)
    file="build_config.py"
    start = "'date':"
    end_symbol = "'end_date':"
    fin = open(file)
    fout = open('tmp_config.py',"w")
    for line in fin:
        if start in line:
            line = line.replace(line.split(':',1)[1].split(',',1)[0],"'"+start_date+"'")
            print(line)
        if end_symbol in line:
            line = line.replace(line.split(':',1)[1].split(',',1)[0],"'" + end + "'")
            print(line)
        fout.write(line)
    fin.close()
    fout.close()
    fin = open('tmp_config.py')
    fout = open(file,'w')
    for line in fin:
        fout.write(line)
    fin.close()
    fout.close()

    os.system("python3 build_train_data.py")
    os.system("python3 stockModel.py 0050")


    stock_symbol = "0050"

    evaluate = Evaluate(stock_symbol)
    finput.write(start_date+"~"+end+"\n")
    finput.write("roi of predict: "+str(evaluate.roi("predict"))+"%\n")
    finput.write("roi of ans: "+str(evaluate.roi("ans"))+"%\n")
    finput.write("roi of baseline: "+str(evaluate.roi("baseline"))+"%\n")
    finput.write("trend_accurancy_rate: "+str(evaluate.trend_accurancy_rate())+"%\n")
