import matplotlib.pyplot as plt
import numpy as np
# 构建数据
x_data = ['accuracy', 'recall', 'F1-score']
y_data = [0.87, 0.82, 0.84]
# 绘图
plt.bar(x=x_data, height=y_data, label='C语言基础', color='indianred', alpha=0.8)
# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
for x, y in enumerate(y_data):
    plt.text(x, y + 100, '%s' % y, ha='center', va='bottom')
    plt.text(x, y + 100, '%s' % y, ha='center', va='top')
# 设置标题
#plt.title("")
# 为两条坐标轴设置名称
# plt.xlabel("年份")
# plt.ylabel("销量")
# 显示图例
#plt.legend()
plt.show()