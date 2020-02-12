#导入必要的模块   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
# Reading data from web
data_url = "diabetes.csv"
df = pd.read_csv(data_url)

#产生测试数据   
#x = np.arange(1,10)
x = df.ix[:, 0]
#y = x
y = df.ix[:, 1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
#设置标题   
ax1.set_title('today')
#设置X轴标签   
plt.xlabel('t')
#设置Y轴标签   
plt.ylabel('r')
#画散点图   
ax1.scatter(x,y,c = 'r',marker = 'o')
#设置图标   
plt.legend('x1')
#显示所画的图   
plt.show()
