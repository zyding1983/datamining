import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 用于线性回归
from sklearn.model_selection import train_test_split

df = pd.read_csv('pm25_train.csv')
df2= pd.read_csv('pm25_test.csv')
def data_format(dt):
    time_list = []
    t = time.strptime(dt, '%Y-%m-%d')
    time_list.append(t.tm_year)
    time_list.append(t.tm_mon)
    time_list.append(t.tm_mday)
    time_list.append(t.tm_wday)
    return time_list
date = df['date'].tolist()
date2= df2['date'].tolist()
jieguo = []
for dt in date:
    jieguo.append(data_format(dt=dt))
jieguo2 = []
for dt2 in date2:
    jieguo2.append(data_format(dt=dt2))

df_time2=pd.DataFrame(jieguo2)
df_time2.columns=['year', 'mon', 'day','week']
df_data2 = pd.concat([df2,df_time2], axis=1)
df_time = pd.DataFrame(jieguo)
df_time.columns=['year', 'mon', 'day','week']
df_data = pd.concat([df,df_time], axis=1)

X_test = df_data2.drop(columns=['date','Ir','cbwd_cv'])
y = df_data['pm2.5']
X = df_data.drop(columns=['pm2.5','date','Ir','cbwd_cv'])
lr = LinearRegression().fit(X, y)

yuce1 = lr.predict(X_test)
te = pd.DataFrame(yuce1)

te.to_csv('a.csv', sep=',', header=True, index=True)

print(yuce1)
'''df_jieguo = pd.DataFrame(y_test)
df_jieguo = df_jieguo.reset_index()
df_jieguo['yuce'] = yuce
df_jieguo['wucha'] = pow((df_jieguo['yuce']-df_jieguo['pm2.5']), 2)
he = sum(df_jieguo['wucha'])

score = he/len(df_jieguo)'''


