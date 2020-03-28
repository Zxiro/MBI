
import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
from evaluate import Evaluate
if len(sys.argv) < 2 :
    stock_symbol = input('輸入股票號碼:')
else:
    stock_symbol = sys.argv[1]
evaluate = Evaluate(stock_symbol)
evaluate.roi()
evaluate.nextweek_predict()
evaluate.predictplt()
evaluate.baseline()
evaluate.accurancy_rate()
evaluate.trend_accurancy_rate()
