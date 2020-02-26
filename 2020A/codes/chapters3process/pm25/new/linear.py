#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pickle
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from sklearn import cross_validation

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
    pm25 = train.loc[:, 'pm2.5'].values
    featuredata = train.drop('pm2.5', axis=1, inplace=False)
    train_x = featuredata.copy()
    train_y = pm25.copy()
    test_x = test.copy()
    # train_x.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_x_dealt.csv", index=False)
    # test_x.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\test_x_dealt.csv", index=False)
    train_x = train_x.values
    test_x = test_x.values
    print('是否存在空缺值？\n',np.any(train.isnull())==True)#不存在空缺值
    return train_x, train_y, test_x

def divid_dataset_testingdata():
    x = joblib.load('F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_x.pkl')
    y = joblib.load('F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_y.pkl')
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)  # 划分
    return train_x, test_x, train_y, test_y

def showseries(series):#时间序列可视化
    plt.figure()
    plt.plot(np.arange(len(series)), series)
    plt.show()
    return 0

def minmaxscale(series):#归一化
    series_scale = preprocessing.minmax_scale(series, feature_range=(0, 1))
    return series_scale

def pca_reduce(train,test):#PCA降维
    line_train=train.shape[0]
    line_test = test.shape[0]
    dataset=np.vstack((train,test))#取交集
    print('dataset=\n', dataset.shape, '\n', dataset)
    pca = decomposition.PCA(n_components=6)
    reduced_dataset = pca.fit_transform(dataset)
    print('dataset_PCA=\n', reduced_dataset.shape, '\n', reduced_dataset)

    plt.figure()
    plt.plot(pca.explained_variance_, 'k', linewidth=2)
    plt.xlabel('n_components', fontsize=16)
    plt.ylabel('explained_variance_', fontsize=16)
    plt.show()

    train=reduced_dataset[range(0,line_train,1)]
    test = reduced_dataset[range(line_train,reduced_dataset.shape[0],1)]
    return train,test


def model_struct(train_x, train_y):#建立模型并储存
    model = LinearRegression()
    model.fit(train_x, train_y)
    with open("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\regs.pickle", 'wb') as f:
        pickle.dump(model, f)
        return model

def prediction(model,test_x):#预测结果并输出到文件夹
    ##预测
    print('test_x=\n', test_x)
    test_y = model.predict(test_x)
    print('test_y=\n', test_y)
    ##输出结果到csv
    result = pd.DataFrame({'pm2.5': test_y})
    result.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\result5_regs.csv", index=False)
    return test_y

def BPNN(train_x,train_y,test_x): #神经网络的构造与预测
    ##构造神经网络
    n = 32
    input_dim = train_x.shape[1]
    output_dim = 1
    model = Sequential(
        [Dense(n, input_dim=input_dim), Activation('relu'), Dense(output_dim=output_dim), Activation('relu')])
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
    ##训练网络
    model.fit(train_x, train_y, nb_epoch=10)
    ##预测结果
    test_y_predict = model.predict(test_x)
    test_y_predict = test_y_predict.reshape(6011, )  # 前面那样的格式好像不能生成字典而导出到csv，调整一下格式
    ##输出结果到csv
    result2 = pd.DataFrame({'pm2.5': test_y_predict}, index=[0])
    result2.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\result2.csv", index=False)
    return test_y_predict

def fault_judge(Name,train):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()  # 建立图像
    data = train[[Name]]
    p = data.boxplot(return_type='dict')  # 画箱线图，直接使用DataFrame的方法
    x = p['fliers'][0].get_xdata()  # 'fliers'即为异常值的标签
    y = p['fliers'][0].get_ydata()
    y.sort()  # 从小到大排序，该方法直接改变原对象
    # 用annotate添加注释
    # 其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
    # 以下参数都是经过调试的，需要具体问题具体调试。
    for i in range(len(x)):
        if i > 0:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / (y[i] - y[i - 1]), y[i]))
        else:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.08, y[i]))
    plt.show()
    return 0
#--------------主函数----------------：
#导入数据集，分割成训练集和测试集，将时间戳特征分隔为年月日
train_x, train_y, test_x = divid_dataset_realdata()
# #异常检测
# fault_judge('cbwd_NE',train_x)
#归一化
train_x=minmaxscale(train_x)
test_x=minmaxscale(test_x)
#pca降维
train_x,test_x = pca_reduce(train_x,test_x)

# ##储存预处理后的数据
# joblib.dump(train_x, 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_x.pkl')
# joblib.dump(train_y, 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_y.pkl')
# joblib.dump(test_x, 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\test_x.pkl')

# # train_x, test_x, train_y, test_y=divid_dataset_testingdata()
#训练多元线性回归模型
model=model_struct(train_x, train_y)

#预测结果并评价
test_y_predict = prediction(model, test_x)
#
# ##将结果画出来
# plt.figure()
# length=100
# plt.plot(np.arange(length), test_y_predict[:length],'b')
# plt.plot(np.arange(length), test_y[:length],'g')
# plt.show()
#--------------主函数----------------：


