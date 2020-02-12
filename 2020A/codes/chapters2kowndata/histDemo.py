import numpy as np
import pylab as pl
#data = np.random.normal(5.0, 3.0, 1000)

import pandas as pd
import matplotlib.pyplot as plt
# Reading data from web
data_url = "diabetes.csv"
df = pd.read_csv(data_url)

y = df.ix[:, 2]
print(y)
pl.hist(y)

pl.xlabel('data')
pl.show()
