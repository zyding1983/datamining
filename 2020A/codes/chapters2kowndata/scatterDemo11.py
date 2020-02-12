import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading data from web
data_url = "iris.csv"
df = pd.read_csv(data_url)

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


N = 135#假设有8个点
#x=[1,2,3,4,5,6,7,8]
x = df.ix[:, 1]
#y=[1,2,3,4,5,6,7,8]
y = df.ix[:, 2]

#c=[1,2,2,2,3,3,4,4]#有4个类别，标签分别是1，2，3，4

c=df.ix[:, 4]#有4个类别，标签分别是1，2，3，4

#m = {1:'o',2:'s',3:'D',4:'+'}
m = {0:'o',1:'s',2:'D'}

cm = list(map(lambda x:m[x],c))#将相应的标签改为对应的marker
print(cm)

fig, ax = plt.subplots()

scatter = mscatter(x, y, c=c, m=cm, ax=ax,cmap=plt.cm.RdYlBu)

plt.show()
