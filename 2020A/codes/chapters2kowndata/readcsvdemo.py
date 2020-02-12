import pandas as pd
import matplotlib.pyplot as plt
# Reading data from web
data_url = "statisticsdata.csv"
df = pd.read_csv(data_url)

#print(df.head())
#print(df.tail())
#print(df.columns)
#print(df.index)
#print(df.T)#数据转置使用T方法
#print(df.ix[:, 0].head())
#数据第一列的前5行
#Python的索引是从0开始而非1。为了取出从11到20行的前3列数据
#print(df.ix[10:20, 0:3])
#print(df.describe())

#y = df.ix[:, 0:8]

#print(y)

plt.show(df.plot(kind='box'))



