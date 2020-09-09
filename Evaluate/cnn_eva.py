import matplotlib.pyplot as plt
import math
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from build_config import stock_dic
from sklearn.metrics import mean_squared_error #均方誤差
from sklearn.metrics import mean_absolute_error #平方絕對誤差
from sklearn.metrics import r2_score#R square

'''引入資料'''
stock = '0050'
x_test = np.load('../StockData/TrainingData/NormtestingX_'+stock+'.npy')
y_test = np.load('../StockData/TrainingData/testingY_'+stock+'.npy')
x_test_ = np.load('../StockData/TrainingData/NormtrainingX_'+stock+'.npy')
y_test_mon = np.load('../StockData/TrainingData/testingY_mon_'+stock+'.npy')
y_test_fri = np.load('../StockData/TrainingData/testingY_fri_'+stock+'.npy')
y_test_mon_ans = np.load('../StockData/TrainingData/trainingY_mon_'+stock+'.npy')
y_test_fri_ans = np.load('../StockData/TrainingData/trainingY_fri_'+stock+'.npy')
'''
def turn_to_bin(list):
    for i in range(len(list)):
        if(list[i]>=0):
            list[i] = 1
        else:
            list[i] = 0
    print(list)

turn_to_bin(y_test)'''

model_dif = load_model('./stockModel/stockmodel_cnn_0050_dif.h5') #引入訓練完model
model_mon = load_model('./stockModel/stockmodel_cnn_0050_mon.h5')
model_fri = load_model('./stockModel/stockmodel_cnn_0050_fri.h5')

model_in_dif = load_model('./stockModel/stockmodel_inception_cnn_0050_dif.h5') #引入訓練完model
model_in_mon = load_model('./stockModel/stockmodel_inception_cnn_0050_mon.h5')
model_in_fri = load_model('./stockModel/stockmodel_inception_cnn_0050_fri.h5')

#print(x_test)
predict_dif = model_dif.predict(x_test_)
predict_f_m = model_fri.predict(x_test_) - model_mon.predict(x_test_)
predict_dif_in = model_in_dif.predict(x_test_)
predict_f_m_in = model_in_fri.predict(x_test_) - model_in_mon.predict(x_test_)
'''print(predict_dif)
print(predict_f_m)
print(y_test)'''
#print(predict_dif_in)


'''MSE = mean_squared_error(y_test, predict)
MAE = mean_absolute_error(y_test, predict)
R2S = r2_score(y_test, predict)'''


ax1 = plt.subplot(2, 1, 1)
plt.plot(y_test_mon_ans, color = 'green', label = 'mon')
plt.plot(model_mon.predict(x_test_), color = 'gray', label = 'predict mon')
plt.plot(y_test_fri_ans, color = 'purple', label = 'fri')
plt.plot(model_in_fri.predict(x_test_), color = 'orange', label = 'predict fri')
plt.legend()
ax2 = plt.subplot(2, 1, 2)
plt.plot(predict_dif, color = 'black',label = 'single model')
plt.plot(predict_f_m , color = 'blue' , label = 'two model')
plt.plot(y_test, color = 'red',label = 'ans')
plt.ylabel('CNN')
plt.legend()

plt.savefig("../../public_html/stock_evaluate/cnn/0050_"+ time.strftime("%m-%d %H-%m-%s", time.localtime())+".png")
plt.clf()

ax3 = plt.subplot(2, 1, 2)
plt.plot(predict_dif_in, color = 'black', label = 'single model')
plt.plot(predict_f_m_in , color = 'blue',label = 'two model' )
plt.plot(y_test, color = 'red', label = 'ans')
plt.legend()
ax4 = plt.subplot(2, 1, 1)
plt.plot(y_test_mon_ans, color = 'green', label = 'mon')
plt.plot(model_in_mon.predict(x_test_), color = 'gray', label = 'predict mon')
plt.plot(y_test_fri_ans, color = 'purple', label = 'fri')
plt.plot(model_fri.predict(x_test_), color = 'orange', label = 'predict fri')
plt.ylabel('INCEPTION')
plt.legend()
plt.savefig("../public_html/stock_evaluate/inception/0050_"+time.strftime("%m-%d %H-%m-%s", time.localtime())+".png")

'''print(MSE)
print(MAE)
print(R2S) # <0 means terrible '''
'''plt.axhline(y = MSE, color='black', linestyle='-', label = 'MSE')
plt.axhline(y = MAE, color='yellow', linestyle='-', label = 'MAE')
plt.axhline(y = R2S, color='green', linestyle='-', label = 'R2S')'''


