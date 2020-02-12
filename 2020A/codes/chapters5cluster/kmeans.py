import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
#data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
#假如我要构造一个聚类数为3的聚类器
data_url = "iris.csv"
df = pd.read_csv(data_url)
data = df.T[0:4].T
#print(data)
estimator = KMeans(n_clusters=3)#构造聚类器
result = estimator.fit_predict(data)
print(result)
