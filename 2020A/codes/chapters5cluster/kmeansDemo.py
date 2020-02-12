from sklearn.cluster import KMeans
import numpy as np
#X = np.array([[1, 2], [1, 4], [1, 0],
#              [10, 2], [10, 4], [10, 0]])

import pandas as pd
from pandas import DataFrame
data_url = "diabetes.csv"
df = pd.read_csv(data_url)
x = df.ix[:, 1:9]
#print(df)

kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
print(kmeans.labels_)

#print(kmeans.predict([[0, 0], [12, 3]]))

#print(kmeans.cluster_centers_)