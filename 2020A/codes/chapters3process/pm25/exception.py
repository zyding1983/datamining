import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
datafile = 'data/pm25_train.csv'
resultfile = 'data/explore.csv'
data = pd.read_csv(datafile)
explore = data.describe(percentiles=[],include='all')
# explore = explore[['null','max','min']]
# explore.columns = ['kongzhi','maxnum','minnum']
#explore.to_csv(resultfile)
data1 = pd.read_csv(datafile,index_col='date')
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
p= data1.boxplot()
x= p['fliers'][0].get_xdata()
y = p['fliers'][0].get_ydata()
y.sort()
for i in range (len(x)):
    if i>0:
        plt.annotate(y[i],xy = (x[i],y[i]),xytext = (x[i]+0.5-0.8/(y[i]-y[i-1]),y[i]))
    else:
        plt.annotate(y[i],xy=(x[i],y[i]),xytest=(x[i]+0.8,y[i]))
plt.show()