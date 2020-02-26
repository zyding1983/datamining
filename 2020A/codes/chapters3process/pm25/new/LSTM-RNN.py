#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import warnings
import pandas as pd
import numpy as np
import datetime
import time
from matplotlib import pyplot
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.backend import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
# ————————————————
# 版权声明：本文为CSDN博主「浅笑古今」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/u012735708/article/details/82769711

def divid_dataset_realdata():#导入数据集，分割成训练集和测试集
    train = pd.read_csv('F:\文档\课程\计算机辅助系统工程\DataMiningProject-PM2.5\data\pm25_train.csv', sep=',', encoding='utf-8')
    test = pd.read_csv('F:\文档\课程\计算机辅助系统工程\DataMiningProject-PM2.5\data\pm25_test.csv', sep=',', encoding='utf-8')

    # train改变时间戳
    current_date = train.loc[:, 'date'].values
    month = np.zeros(train.shape[0])
    day = np.zeros(train.shape[0])
    year = np.zeros(train.shape[0])
    week = np.zeros(train.shape[0])
    for i in range(current_date.shape[0]):
        tmp_date_list = current_date[i].split('-')
        timeArray = time.strptime(str(tmp_date_list[0]), "%Y/%m/%d")
        month[i] = int(timeArray.tm_mon)
        day[i] = int(timeArray.tm_mday)
        year[i] = int(timeArray.tm_year)
        week[i] = int(datetime.datetime(int(year[i]), int(month[i]), int(day[i])).strftime("%w"))
    train.insert(13, 'month', month)
    train.insert(14, 'day', day)
    train.insert(15, 'year', year)
    train.insert(16, 'week', week)
    # test改变时间戳
    current_date = test.loc[:, 'date'].values
    month = np.zeros(test.shape[0])
    day = np.zeros(test.shape[0])
    year = np.zeros(test.shape[0])
    week = np.zeros(test.shape[0])
    for i in range(current_date.shape[0]):
        tmp_date_list = current_date[i].split('-')
        timeArray = time.strptime(str(tmp_date_list[0]), "%Y/%m/%d")
        month[i] = int(timeArray.tm_mon)
        day[i] = int(timeArray.tm_mday)
        year[i] = int(timeArray.tm_year)
        week[i] = int(datetime.datetime(int(year[i]), int(month[i]), int(day[i])).strftime("%w"))
    test.insert(12, 'month', month)
    test.insert(13, 'day', day)
    test.insert(14, 'year', year)
    test.insert(15, 'week', week)
    test = test.drop('date', axis=1, inplace=False)
    train = train.drop('date', axis=1, inplace=False)

    print(train.columns)
    print(test.columns)
    title=test.columns
    pm25 = train.loc[:, 'pm2.5'].values
    featuredata = train.drop('pm2.5', axis=1, inplace=False)
    train_x = featuredata.copy()
    train_y = pm25.copy()
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(train_x, train_y, test_size=0.3, random_state=0)  # 划分
    test_x_real = test.values
    # train_x.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_x_dealt.csv", index=False)
    # test_x.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\test_x_dealt.csv", index=False)
    train_x = train_x.values
    test_x = test_x.values
    print('是否存在空缺值？\n',np.any(train.isnull())==True)#不存在空缺值
    return train_x, train_y, test_x, test_y, title, test_x_real

#--------------主函数----------------：
#导入数据集，分割成训练集和测试集，将时间戳特征分隔为年月日
train_x, train_y, test_x, test_y, title, test_x_real = divid_dataset_realdata()
# values = train_x
# groups = [1, 2, 3, 5, 6]
# i=1
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[0:1000, group])
#     pyplot.title(title[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()

#归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_x=scaler.fit_transform(train_x)
test_x=scaler.fit_transform(test_x)
test_x_real=scaler.fit_transform(test_x_real)
test_y=scaler.fit_transform(test_y)
train_y=scaler.fit_transform(train_y)
# #pca降维
#
##改造数据形式为适合LSTM的格式
train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
test_X_real = test_x_real.reshape((test_x_real.shape[0], 1, test_x_real.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape,test_y.shape)

#搭建LSTM
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

##预测
test_y_pred = model.predict(test_X_real)
temp=test_X_real.reshape((test_X_real.shape[0], test_X_real.shape[2]))#转换成原格式
# temp = temp.astype(np.float32)
# inv_yhat = concatenate((test_y_pred, temp[:, 1:]), axis=1)
PRED = scaler.inverse_transform(test_y_pred)
PRED=PRED.reshape((PRED.shape[0],))
##输出结果到csv
result = pd.DataFrame({'pm2.5': PRED})
result.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\result6_LSTM.csv", index=False)

# ##检测模型对于已有数据集的预测效果
# yhat = model.predict(test_X)
# test_X =test_X.reshape((test_X.shape[0], test_X.shape[2]))#转换成原格式
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)

