import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
import pandas as pd
data_url = "diabetes.csv"
df = pd.read_csv(data_url)
x = df.ix[:, 0:8]
pca = PCA(n_components=8)
pca.fit(x)


print(pca.explained_variance_ratio_)

#print(pca.transform(X) )



