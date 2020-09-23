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
    number_of_data = int(end_year) - int(start_year) - 10 + 1
    if(number_of_data <= 0):
        number_of_data = 1
    #print(number_of_data)

load_csv('0050')

#finput = open("ten_year_evaluate.csv","w")
finput = open("../public_html/tenyear_eva.csv","w")
finput.write("year"+",")
finput.write("roi of predict"+",")
finput.write("roi of ans"+",")
finput.write("roi of baseline,")
finput.write("trend_accurancy_rate_train"+",")
finput.write("trend_accurancy_rate_val"+",")
finput.write("trend_accurancy_rate_test"+"\n")
print(start_year)
print(end_date)
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
    #os.system("python3 stockModel.py 0050")
    train = 0
    test = 0
    val = 0
    predict = 0
    ans = 0
    baseline = 0
    stock_symbol = "0050"
    for i in range(5):
        os.system("python3 model_start.py 0050")
        evaluate = Evaluate(stock_symbol)
        predict += evaluate.roi("predict")
        ans += evaluate.roi("ans")
        baseline += evaluate.roi("baseline")
        train += (evaluate.trend_accurancy_rate("train"))
        test += (evaluate.trend_accurancy_rate("test"))
        val += (evaluate.trend_accurancy_rate("val"))
    #os.system("python3 ./Model/transformer.py 0050")
    train /= 5
    test /= 5
    val /= 5
    baseline /= 5
    predict /= 5
    ans /= 5

    #stock_symbol = "0050"

    evaluate = Evaluate(stock_symbol)
    finput.write(start_date+"~"+end+",")
    finput.write(str(predict)+"%,")
    finput.write(str(ans)+"%,")
    finput.write(str(baseline)+"%,")
    finput.write(str(train)+"%,")
    finput.write(str(val)+"%,")
    finput.write(str(test)+"%\n")
    finput.write("\n")
    evaluate.predictplt()

finput.close()
