import numpy as np
from sklearn.preprocessing import Imputer
import pandas as pd
data_url = "diabetes.csv"
df = pd.read_csv(data_url)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

#imp.fit([[1, 2], [np.nan, 3], [7, 6]])
imp.fit(df)

#X = [[np.nan, 2], [6, np.nan], [7, 6]]

print(imp.transform(df))
