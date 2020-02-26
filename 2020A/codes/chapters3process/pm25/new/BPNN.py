from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

##读取数据,将训练集分成训练集和测试集，来检验算法精度
# test_x=joblib.load( 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\test_x.pkl')
# test_y=joblib.load( 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\test_y.pkl')
x=joblib.load( 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_x.pkl')
y=joblib.load( 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\train_y.pkl')
# ##还是要对y进行下归一化
# y = mm.fit_transform(y)

# train_x,test_x,train_y,test_y =cross_validation.train_test_split(x, y, test_size=0.0, random_state=0)#划分

train_x=x.copy()
train_y=y.copy()
test_x=joblib.load( 'F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\test_x.pkl')
##构造神经网络
activation1 = 'Relu'
activation2 = 'tanh'
n = 30
input_dim = train_x.shape[1]
output_dim = 1
model = Sequential([Dense(n, input_dim=input_dim), Dense(output_dim=output_dim)])
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])

##训练网络
model.fit(train_x, train_y, nb_epoch=100)

##预测结果
# loss, accuracy = model.evaluate(test_x,test_y)
test_y_prediction = model.predict(test_x)
test_y_prediction = test_y_prediction.reshape(test_y_prediction.shape[0],)  # 前面那样的格式好像不能生成字典而导出到csv，调整一下格式
# # #反归一化
# test_y_prediction2 = mm.inverse_transform(test_y_prediction)
##将预测结果输出
result3 = pd.DataFrame({'pm2.5': test_y_prediction})
result3.to_csv("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\result5_nonenone.csv", index=False)

# ##将结果画出来
# plt.figure()
# length=100
# plt.plot(np.arange(length), test_y_prediction[:length],'b')
# plt.plot(np.arange(length), test_y[:length],'g')
# plt.title('loss=%f,accuracy=%f '%(loss, accuracy)) # 对图形整体增加文本标签
# plt.savefig("F:\\文档\\课程\\计算机辅助系统工程\\回归分析\\我的模型\\神经网络调参\\"+activation1+"_"+activation2+"_points="+str(n)+".png")
# plt.show()
