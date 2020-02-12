import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#matplotlib画图中中文显示会有问题，需要这两行设置默认字体

plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=9,xmin=0)
plt.ylim(ymax=9,ymin=0)
#画两条（0-9）的坐标轴并设置轴标签x，y

import pandas as pd

# Reading data from web
data_url = "iris11.csv"
df = pd.read_csv(data_url)

data_url1 = "iris222.csv"
df1 = pd.read_csv(data_url1)

#x1 = np.random.normal(2,1.2,300) # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标

#y1 = np.random.normal(2,1.2,300) # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标

x1 = df.ix[:, 1]
#y=[1,2,3,4,5,6,7,8]
y1 = df.ix[:, 2]

x2 = df1.ix[:, 1]
#y=[1,2,3,4,5,6,7,8]
y2 = df1.ix[:, 2]

#x2 = np.random.normal(7.5,1.2,300)
#y2 = np.random.normal(7.5,1.2,300)
colors1 = '#00CED1' #点的颜色
colors2 = '#DC143C'
area = np.pi * 4**2  # 点面积 
# 画散点图
plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')
plt.plot([0,9.5],[9.5,0],linewidth = '0.5',color='#000000')
plt.show()