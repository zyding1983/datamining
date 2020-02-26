#coding=utf-8
# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from sklearn import datasets, linear_model

def main():
    pass

def get_data(file_name):
    data = pd.read_csv(file_name)

    return data
    # X_parameter = []
    # Y_parameter = []
    # for single_square_feet ,single_price_value in zip(data['square_feet'],data['price']):
    #     X_parameter.append([float(single_square_feet)])
    #     Y_parameter.append(float(single_price_value))
    # return X_parameter,Y_parameter

data = get_data("data/pm25_train.csv")
X = data[['date','hour','DEWP','TEMP','PRES','Iws','Is','Ir','cbwd_NE','cbwd_NW','cbwd_SE','cbwd_cv','pm2.5']]
Y = data['pm2.5']
data["date"] = pd.to_datetime(data["date"])
plt.xlabel("Month")    #x、y轴标签、标题都不支持中文
plt.ylabel("Unemployment Rate")
data["MONTH"] = data["date"].dt.month #增加一列 月份 值：1-12
#plt.plot(data[0:12000]["MONTH"],data[0:12000]["pm2.5"],c = "red")
date = X['date'].copy()
month = np.zeros(X.shape[0])
day = np.zeros(X.shape[0])
year =  np.zeros(X.shape[0])
week =  np.zeros(X.shape[0])
print (date)
for i,val in enumerate(date):
    month[i] = int(val[5:7])
    day[i] = int(val[8:])
    year[i] = int(val[0:4])
    week[i] = datetime.datetime(int(val[0:4]),int(val[5:7]),int(val[8:])).strftime("%w")
X.insert(13,'month',month)
X.insert(14,'day',day)
X.insert(15,'year',year)
X.insert(16,'week',week)


plt.figure(figsize=(10,20),dpi=88)
plt.subplot(7,1,1)
plt.xlabel("hour")
plt.ylabel("pm2.5")
t1 = X[['hour','pm2.5']].groupby('hour').mean()
plt.scatter(t1.index,t1['pm2.5'])
# plt.scatter(X['PRES'],Y)


plt.subplot(7,1,2)
plt.xlabel("DEWP")
plt.ylabel("pm2.5")
t1 = X[['DEWP','pm2.5']].groupby('DEWP').mean()
plt.scatter(t1.index,t1['pm2.5'])

plt.subplot(7,1,3)
plt.xlabel("TEMP")
plt.ylabel("pm2.5")
t2 = X[['TEMP','pm2.5']].groupby('TEMP').mean()
plt.scatter(t2.index,t2['pm2.5'])

plt.subplot(7,1,4)
plt.xlabel("Iws")
plt.ylabel("pm2.5")
t1 = X[['Iws','pm2.5']].groupby('Iws').mean()
plt.scatter(t1.index,t1['pm2.5'])

plt.subplot(7,1,5)
plt.xlabel("Is")
plt.ylabel("pm2.5")
t1 = X[['Is','pm2.5']].groupby('Is').mean()
plt.scatter(t1.index,t1['pm2.5'])

plt.subplot(7,1,6)
plt.xlabel("week")
plt.ylabel("pm2.5")
t1 = X[['week','pm2.5']].groupby('week').mean()
plt.scatter(t1.index,t1['pm2.5'])
plt.subplots_adjust(hspace=0.5)

plt.subplot(7,1,7)
plt.xlabel("cbwd_NW")
plt.ylabel("pm2.5")
t1 = X[['cbwd_NW','pm2.5']].groupby('cbwd_NW').mean()
plt.scatter(t1.index,t1['pm2.5'])
plt.subplots_adjust(hspace=0.5)


plt.show()


if __name__ == '__main__':
  main()