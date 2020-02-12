from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pandas as pd
data_url = "iris.csv"
df = pd.read_csv(data_url)
x = df.ix[:, 1:5]
y = df.ix[:, 5]

clf = GaussianNB()
clf = clf.fit(x, y)

data_urltest = "iristest.csv"
dftest = pd.read_csv(data_urltest)
xtest = dftest.ix[:, 1:5]

print(clf.predict(xtest))