import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
data_train = pd.read_csv("train_feature.csv")  # 训练数据
data_train_y = pd.read_csv("train_label.csv")  # 训练标签
data_test = pd.read_csv("test_feature.csv")  # 测试数据

# 对训练集reshape 8时刻合一天 ，维度增加到64维，最后再清洗去掉重复列
train_feature = data_train.values.ravel().reshape(-1)
print(train_feature, '\n shape :', train_feature.shape, '=', 17008 * 8)
feature_train_64 = train_feature.reshape([2126, 64])
print(feature_train_64, '\nshape :', feature_train_64.shape)

# 对测试集reshape 8时刻合一天，维度增加到64维，最后再清洗去掉重复列
test_feature = data_test.values.ravel().reshape(-1)
print(test_feature, '\n shape :', test_feature.shape, '=', 7320 * 8)
feature_test_64 = test_feature.reshape([915, 64])
print(feature_test_64, '\nshape :', feature_test_64.shape)

# 数据简单的清洗
feature_name = data_train.columns.tolist()
feature_name.remove('日期')
time = data_train['时刻'][0:8].tolist()
feature_list = []
for j in time:
    feature_list.append('日期')
    for i in feature_name:
        fn = i + '_' + str(j)
        feature_list.append(fn)

len(feature_list)
print(feature_list)

# 对训练集转换为DataFrame
# 转换成DataFrame类型的数据
data_train = pd.DataFrame(feature_train_64, columns=feature_list)
# 去除重复列
columns = data_train.columns.tolist()
data_train_deal = data_train[columns].T.drop_duplicates().T
data_train_deal.head()
data_train.head()

# 合并数据集，将变量和目标变量合并
data_train_final = pd.concat([data_train_deal, data_train_y['电场实际太阳辐射指数']], axis=1, join='inner')

# 对测试集转换为DataFrame
data_test_deal = pd.DataFrame(feature_test_64, columns=feature_list)
# 将重复列去掉
columns = data_test_deal.columns.tolist()
data_test_final = data_test_deal[columns].T.drop_duplicates().T
data_test_final.head()

# 划分数据集
X = np.array(data_train_final.drop(['电场实际太阳辐射指数'], axis=1))
y = np.array(data_train_y['电场实际太阳辐射指数'])
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_test2 = np.array(data_test_final)
print(X_test2.shape)

# 数据标准化
# 标准化
scaler_train = StandardScaler().fit(X)
X_train_std = scaler_train.transform(X)
print(X_train_std.shape)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_train_std, y, test_size=0.3, random_state=45)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 对测试集进行标准化
scaler_test = StandardScaler().fit(X_test2)
X_test_std = scaler_test.transform(X_test2)
print(X_test_std.shape)
#使用xgboost
from xgboost import XGBRegressor
reg1 = XGBRegressor(max_depth = 6,
            learning_rate = 0.02,
            n_estimators = 160,
            silent = True,
            objective = 'reg:linear',
            booster = "gbtree",
            gamma = 0.1,
            min_child_weight = 1,
            subsample = 0.7,
            colsample_bytree = 0.5,
            reg_alpha = 0,
            reg_lamda = 10,
            random_state = 1000)
reg1.fit(X_train_std, y_train_std)
#y_pred_line1 = reg.predict(X_test_std)
# 使用线性模型训练
from sklearn import linear_model
reg2 = linear_model.LinearRegression()
reg2.fit(X_train_std, y_train_std)
#RandomForestRegressor
from sklearn.ensemble.forest import RandomForestRegressor
reg3 =  RandomForestRegressor()
reg3.fit(X_train_std, y_train_std)
#y_pred_line2 = reg.predict(X_test_std)
#mae_line = mean_absolute_error(y_pred_line, y_test)
#print("MAE line_score:", mae_line)

# 导出数据
index = np.array(data_test_final['日期']).astype("int32")
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
result_lgb = reg1.predict(X_test_std)
result_xgb = reg2.predict(X_test_std)
result_rfr = reg3.predict(X_test_std)
y_hat_line = ( 4*result_lgb+  5* result_xgb+1*result_rfr)/10
result = pd.DataFrame({"time": index, "prediction": y_hat_line})
columns = ["time", "prediction"]
result = result.loc[:, columns]
result.to_csv("output/submit_baseline_line{}.csv".format(now), index=False)