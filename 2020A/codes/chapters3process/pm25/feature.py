import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,Ridge,LinearRegression,RandomizedLogisticRegression
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
def load_data():
    df_train = pd.read_csv("data/pm25_train.csv")
    df_test = pd.read_csv("data/pm25_test.csv")
    del df_train['date']
    del df_test['date']
    # del df_train['TEMP']
    # del df_train['cbwd_NE']
    print(df_train.corr()[['pm2.5']].sort_values('pm2.5'))
    prices = pd.DataFrame({"price": df_train['pm2.5'], "log(price+1)": np.log1p(df_train["pm2.5"])})
    prices.hist()
    y_true = df_train['pm2.5'].values
    min_max_scaler = preprocessing.MinMaxScaler()
    df_train = min_max_scaler.fit_transform(df_train)
    df_test = min_max_scaler.fit_transform(df_test)
    # #df_train = preprocessing.normalize(df_train, norm='l1')
    # df_train = (df_train - np.mean(df_train, axis=0)) / np.std(df_train, axis=0)  # 标准化
    # df_train = preprocessing.scale(df_train)


    # 用二次多项式对样本X值做变换

    return df_train, df_test,y_true
def model():
    df_train, df_test,y_true = load_data()
    # X_train, X_test, y_train, y_test = train_test_split(df_train.drop('pm2.5', axis=1).values,
    #                                                     df_train['pm2.5'].values, test_size=0.3, random_state=0)
    Y = df_train[:,0]
    X = np.delete(df_train,0,1)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
    quadratic_featurizer = preprocessing.PolynomialFeatures(degree=2)
    X_train = quadratic_featurizer.fit_transform(X_train)
    X_test = quadratic_featurizer.fit_transform(X_test)
    df_test = quadratic_featurizer.fit_transform(df_test)
    print(df_train.shape)
    clf = Ridge()
    rfm = RFE(estimator=clf, n_features_to_select=10, step=1)
    rfm.fit(X_train,y_train)
    print(X_train.shape)
    X_new = rfm.transform(X_train)
    X_new1 = rfm.transform(X_test)
    X_new2 = rfm.transform(df_test)
    print(X_new.shape)
    print("333333222222233333")

    # param_test1 = {'n_estimators': range(10, 100, 10)}
    # gsearch1 = GridSearchCV(estimator=RandomForestRegressor(min_samples_split=100,
    #                                                          min_samples_leaf=20, max_depth=8, max_features='sqrt',
    #                                                          random_state=10),
    #                         param_grid=param_test1, scoring='neg_mean_squared_error', cv=10)

    param_test2 = {'max_depth': range(3, 14, 2)}

    print("333444444442233333")

    gsearch2 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=80,min_samples_split=100,
                                                             min_samples_leaf=20, oob_score=True, random_state=10),
                            param_grid=param_test2, scoring='neg_mean_squared_error', iid=False, cv=10)

    print("33335555533333")
    LRModels = gsearch2.fit(X_new, y_train)
    print("333356666666533333")
    print(LRModels.best_params_)
    print("333333333333333")
    #预测模型
    y_pred = np.expm1(LRModels.predict(X_new1))
    y_pred2 = np.expm1(LRModels.predict(X_new2))
    print(y_test)
    print(y_pred)
    print(explained_variance_score(y_test,y_pred,multioutput='raw_values'))
    mse = np.average((y_pred - np.array(y_test)) ** 2)
    print(mse)
    y_pred = y_pred*np.std(y_true)+np.mean(y_true)
    y_pred2 = y_pred2 * np.std(y_true) + np.mean(y_true)
    print(y_pred2.shape)
    print(y_pred2)
    result = pd.DataFrame({'pm2.5':y_pred2.astype(np.int32)})
    result.to_csv("data/result.csv",index=False)
if __name__ == '__main__':
    model()