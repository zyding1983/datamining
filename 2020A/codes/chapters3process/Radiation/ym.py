#a -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import regularizers as kreg
from keras.layers import Input, Add
from keras.models import Model

def readData(fileName, encoding='utf-8'):
    result = []
    with open(fileName, encoding=encoding) as file:
        for line in file:
            line = line.replace('\n', '')
            one = line.split(',')
            result.append(one)    
    return np.asarray(result)



feature = readData('train_feature.csv')
label = readData('train_label.csv')
test = readData('test_feature.csv')



# 删除数据头部
feature = np.delete(feature, [0], axis=0)
feature = np.delete(feature, [6, 7], axis=1)
feature = feature.astype(np.float)

# 删除数据头部
label = np.delete(label, [0], axis=0)
label = np.delete(label, [0], axis=1)
label = label.astype(np.float)
label = label.flatten()

# 删除数据头部
test = np.delete(test, [0], axis=0)
test = np.delete(test, [6, 7], axis=1)
test = test.astype(np.float)

# 取各列的最小值和最大值
feature_col_max = np.max(feature, axis=0) 
feature_col_min = np.min(feature, axis=0) - 0.00001
test_col_max = np.max(test, axis=0) 
test_col_min = np.min(test, axis=0) - 0.00001
# print(col_min)
# exit()

# 归一化特征值
feature = (feature - feature_col_min) / (feature_col_max - feature_col_min)
test = (test - test_col_min) / (test_col_max - test_col_min)


feature = np.reshape(feature, (-1, 48))
test = np.reshape(test, (-1, 48))
# all_feature = (all_feature+0.1) * 0.9
print(feature.size)


input_data = Input(shape=(48,))

dense1 = Dense(48, activation='elu')(input_data)
droupout1 = Dropout(0.5)(dense1)
dense2 = Dense(24, activation='elu')(droupout1)
droupout2 = Dropout(0.5)(dense2)
outputs = Dense(1, activation='elu')(droupout2)
model = Model(inputs=input_data, outputs=outputs)
model.compile(optimizer='adam', loss='mae')

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(feature,label)

model.fit(x=feature, y=label, batch_size=32, epochs=400, verbose=2)

preds = model.predict(test)
preds = preds.flatten()
result = []
index = np.arange(preds.shape[0]) + 1
result.append(index)
result.append(preds)
result = np.asarray(result)
result = np.transpose(result, (1,0))
f = open('lebel_result.csv', 'w')
f.write('time,prediction\n')
for i in range(result.shape[0]):
    f.write(str(int(result[i][0]))+','+str(result[i][1])+'\n')
f.close()