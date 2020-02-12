from sklearn import tree
import pandas as pd
from sklearn.naive_bayes import GaussianNB
data_url = "diabetes.csv"
df = pd.read_csv(data_url)
x = df.ix[:, 0:8]
#print(x)
y= df.ix[:, 8]
#print(y)
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
clf = GaussianNB()
clf = clf.fit(x, y)

data_urltest = "diabetestest.csv"
dftest = pd.read_csv(data_urltest)

print(clf.predict(dftest))
#print(clf.predict_proba([[2., 2.]]))