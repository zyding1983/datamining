from sklearn import preprocessing
import numpy as np
import pandas as pd
from pandas import DataFrame
data_url = "diabetes.csv"
df = pd.read_csv(data_url)
x = df.ix[:, 0:8]
#print(df)
#X_train = np.array([[ 56, 87777,  2],
 #                      [ 45,  76565,  3],
 #                       [ 36,  54545, 5]])

#X_scaled = preprocessing.MinMaxScaler(x)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(x)

print(X_train_minmax)

df1 = DataFrame(X_train_minmax)
#df1 = DataFrame(X_scaled,index= ['1','2','3','4','5','6','7','8'])

df1.to_csv("test.csv")