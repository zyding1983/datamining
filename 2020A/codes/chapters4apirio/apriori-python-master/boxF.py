__author__ = 'lx'
import pandas as pd
data_url = "F:\class\datamining\ppt\class1\data.csv"
df = pd.read_csv(data_url)

import matplotlib.pyplot as plt
plt.show(df.plot(kind = 'box'))

