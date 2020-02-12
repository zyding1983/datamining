__author__ = 'lx'
import numpy as np
from sklearn.decomposition import IncrementalPCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
ipca = IncrementalPCA(n_components=1, batch_size=3)
ipca.fit(X)
IncrementalPCA(batch_size=3, copy=True, n_components=2, whiten=False)
print(ipca.transform(X) )
