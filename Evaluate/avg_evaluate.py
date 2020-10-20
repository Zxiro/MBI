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
    #start_year = stock_data['date'][0].split('-',1)[0]
    #end_year = stock_data.iloc[-1]['date'].split('-',1)[0]
    #start_date = stock_data['date'][0]
    end_date = stock_data.iloc[-1]['date']
    #start
    start_date = end_date.replace(end_date.split('-',1)[0],str(int(end_date.split('-',1)[0]) - 9))
    #print(number_of_data)
load_csv('0050')

#finput = open("ten_year_evaluate.csv","w")
finput = open("../public_html/avg_eva.csv","w")
finput.write("year"+",")
finput.write("roi of predict"+",")
finput.write("roi of ans"+",")
finput.write("roi of baseline,")
finput.write("trend_accurancy_rate_train"+",")
finput.write("trend_accurancy_rate_val"+",")
finput.write("trend_accurancy_rate_test"+"\n")

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
        line = line.replace(line.split(':',1)[1].split(',',1)[0],"'" + end_date + "'")
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
train = np.zeros(5)
test = np.zeros(5)
val = np.zeros(5)
predict = np.zeros(5)
ans = np.zeros(5)
baseline = np.zeros(5)
stock_symbol = "0050"
for i in range(5):
    os.system("python3 model_start.py 0050")
    evaluate = Evaluate(stock_symbol)
    predict[i] = evaluate.roi("predict")
    ans[i] = evaluate.roi("ans")
    baseline[i] = evaluate.roi("baseline")
    train[i] = (evaluate.trend_accurancy_rate("train"))
    test[i] = (evaluate.trend_accurancy_rate("test"))
    val[i] = (evaluate.trend_accurancy_rate("val"))
#os.system("python3 ./Model/transformer.py 0050")
'''train /= 5
test /= 5
val /= 5
baseline /= 5
predict /= 5
ans /= 5
'''
#stock_symbol = "0050"
'''
evaluate = Evaluate(stock_symbol)
finput.write(start_date+"~"+end_date+",")
finput.write(str(np.mean(predict))+"%/"+str(np.std(predict, ddof=1))+"%,")
finput.write(str(np.mean(ans))+"%/"+str(np.std(ans, ddof=1))+"%,")
finput.write(str(np.mean(baseline))+"%/"+str(np.std(baseline, ddof=1))+"%,")
finput.write(str(np.mean(train))+"%/"+str(np.std(train, ddof=1))+"%,")
finput.write(str(np.mean(val))+"%/"+str(np.std(val, ddof=1))+"%,")
finput.write(str(np.mean(test))+"%/"+str(np.std(test, ddof=1))+"%\n")
finput.write("\n")

'''

evaluate = Evaluate(stock_symbol)
finput.write(start_date+"~"+end_date+",")
finput.write(str(np.mean(predict))+"%,")
finput.write(str(np.mean(ans))+"%,")
finput.write(str(np.mean(baseline))+"%,")
finput.write(str(np.mean(train))+"%,")
finput.write(str(np.mean(val))+"%,")
finput.write(str(np.mean(test))+"%\n")
finput.write(",")
finput.write(str(np.std(predict, ddof=1))+"%,")
finput.write(str(np.std(ans, ddof=1))+"%,")
finput.write(str(np.std(baseline, ddof=1))+"%,")
finput.write(str(np.std(train, ddof=1))+"%,")
finput.write(str(np.std(val, ddof=1))+"%,")
finput.write(str(np.std(test, ddof=1))+"%\n")
finput.write("\n")
evaluate.predictplt()

finput.close()
