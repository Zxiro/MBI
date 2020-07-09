import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
from evaluate import Evaluate
from build_config import stock_dic
if len(sys.argv) < 2 :
    stock_symbol = stock_dic['stock_num']
else:
    stock_symbol = sys.argv[1]
evaluate = Evaluate(stock_symbol)
evaluate.roi("predict")
evaluate.roi("ans")
evaluate.roi("baseline")
evaluate.predictplt()
evaluate.nextweek_predict()
evaluate.accurancy_rate()
evaluate.trend_accurancy_rate()
