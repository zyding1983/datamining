import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
#data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
data_url = "iris.csv"
df = pd.read_csv(data_url)
data = df.T[0:4].T
clustering = AgglomerativeClustering(linkage= 'average', n_clusters=3)
result = clustering.fit_predict(data)
print(result)
