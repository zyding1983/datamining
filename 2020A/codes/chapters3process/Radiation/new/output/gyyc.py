# -*- coding: utf-8 -*-
# 蒸汽预测

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import math
import os
import seaborn as sns
import keras.backend as K
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import pca
import xgboost
from xgboost import XGBRegressor
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from scipy import stats
from scipy.stats import norm, skew

seed = 2018


# Stacking
####################################################################################
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(clf)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions
        print(out_of_fold_predictions.shape)
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# 简单模型融合
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # 遍历所有模型，你和数据
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    # 预估，并对预估结果值做average
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        # return 0.85*predictions[:,0]+0.15*predictions[:,1]
        # return 0.7*predictions[:,0]+0.15*predictions[:,1]+0.15*predictions[:,2]
        return np.mean(predictions, axis=1)


def load_train_data():
    df = pd.read_csv("zhengqi_train.txt", header=0, sep="\s+")
    # print(df.describe())
    X = df.drop(columns=["target"])
    y = df["target"]
    print("X shape:", X.shape)
    print("y shape", y.shape)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_val shape:", X_val.shape)
    # print("y_val shape:", y_val.shape)
    # return X_train, X_val, y_train, y_val
    return X, y


def load_test_data():
    df = pd.read_csv("zhengqi_test.txt", header=0, sep="\s+")
    # print(df.describe())
    X_test = df
    return X_test


def build_nn():
    model = Sequential()
    model.add(Dense(units=128, activation='linear', input_dim=18))
    model.add(Dense(units=32, activation='linear'))
    model.add(Dense(units=8, activation='linear'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


def build_model():
    svr = make_pipeline(SVR(kernel='linear'))
    line = make_pipeline(LinearRegression())
    lasso = make_pipeline(Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)
    lgbm = lightgbm.LGBMRegressor(learning_rate=0.01, n_estimators=500, num_leaves=31)
    xgb = xgboost.XGBRegressor(booster='gbtree', colsample_bytree=0.8, gamma=0.1,
                               learning_rate=0.02, max_depth=5,
                               n_estimators=500, min_child_weight=0.8,
                               reg_alpha=0, reg_lambda=1,
                               subsample=0.8, silent=1,
                               random_state=seed, nthread=2)
    nn = KerasRegressor(build_fn=build_nn, nb_epoch=500, batch_size=32, verbose=2)
    return svr, line, lasso, ENet, KRR1, KRR2, lgbm, xgb, nn


def rmsle_cv(model=None, X_train_head=None, y_train=None):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(X_train_head)
    rmse = -cross_val_score(model, X_train_head, y_train, scoring="neg_mean_squared_error", cv=kf)
    return (rmse)


def main():
    # X_train, X_val, y_train, y_val = load_train_data()
    print("Load data from file......")
    X_train, y_train = load_train_data()
    X_test = load_test_data()
    # ntrain = len(X_train)
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("y_train shape", y_train.shape)
    all_data = pd.concat([X_train, X_test])
    print(all_data.shape)
    print("Load done.")
    # 数据观察（可视化）
    # import seaborn
    # seaborn.distplot(y_train)
    # plt.show()
    # for col in all_data.columns:
    #    seaborn.distplot(X_train[col])
    #    seaborn.distplot(X_test[col])
    #    plt.show()
    # 异常值
    all_data = all_data.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1)
    print(all_data.shape)
    # X = X.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1)
    # X_test = X_test.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1)
    print("Drop done.")
    # need_col_list = ["6", "7", "8", "10", "16", "21", "27", "30", "31", "32", "36"]
    # X_train = process_error(X_train, need_col_list)
    # X_test = process_error(X_test, need_col_list)
    # all_data = process_error(all_data, need_col_list)
    # 标准化
    from sklearn import preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data = pd.DataFrame(scaler.fit_transform(all_data), columns=all_data.columns)
    print("Scale done.")
    # print("缩放后的describe", all_data.describe())
    # 偏态处理
    # skewed_feats = all_data.apply(lambda x: skew(x.dropna())).sort_values(ascending=True)
    # print("\nSkew in numerical features: \n", skewed_feats)
    # skewness = pd.DataFrame({'Skew' :skewed_feats})
    # print(skewness.head(15))
    # skewness = skewness[abs(skewness) > 0.75]
    # print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    # from scipy.special import boxcox1p
    # skewed_features = skewness.dropna().index
    # lam = 0.15
    # for feat in skewed_features:
    # all_data[feat] += 1
    #    all_data[feat] = boxcox1p(all_data[feat], lam)
    # all_data = pd.get_dummies(all_data)
    # print(all_data.shape)
    # X_train = all_data[:ntrain]
    # X_test = all_data[ntrain:]
    all_data['V0'] = all_data['V0'].apply(lambda x: math.exp(x))
    all_data['V1'] = all_data['V1'].apply(lambda x: math.exp(x))
    # all_data['V4'] = all_data['V4'].apply(lambda x:math.exp(x))
    all_data['V6'] = all_data['V6'].apply(lambda x: math.exp(x))
    all_data['V7'] = all_data['V7'].apply(lambda x: math.exp(x))
    all_data['V8'] = all_data['V8'].apply(lambda x: math.exp(x))
    # all_data['V12'] = all_data['V12'].apply(lambda x:math.exp(x))
    # all_data['V16'] = all_data['V16'].apply(lambda x:math.exp(x))
    # all_data['V26'] = all_data['V26'].apply(lambda x:math.exp(x))
    # all_data['V27'] = all_data['V27'].apply(lambda x:math.exp(x))
    all_data["V30"] = np.log1p(all_data["V30"])
    # all_data["V31"] = np.log1p(all_data["V31"])
    # all_data["V32"] = np.log1p(all_data["V32"])
    # y_train = np.exp(y_train)
    scaled = pd.DataFrame(preprocessing.scale(all_data), columns=all_data.columns)
    X_train = scaled.loc[0:len(X_train) - 1]
    X_test = scaled.loc[len(X_train):]
    print("y skew:", skew(y_train))
    print("Skewness done.")
    print("偏态后的shape", X_train.shape, X_test.shape, y_train.shape)
    # 数据观察（可视化）
    # import seaborn
    # seaborn.distplot(y_train)
    # plt.show()
    # for col in all_data.columns:
    #    seaborn.distplot(X_train[col])
    #    seaborn.distplot(X_test[col])
    #    plt.show()
    # 特征选择
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    # 方差
    threshold = 0.85
    vt = VarianceThreshold().fit(X_train)
    # Find feature names
    feat_var_threshold = X_train.columns[vt.variances_ > threshold * (1 - threshold)]
    X_train = X_train[feat_var_threshold]
    X_test = X_test[feat_var_threshold]
    all_data = pd.concat([X_train, X_test])
    print("方差后的shape", all_data.shape)
    # 单变量
    X_scored = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)
    feature_scoring = pd.DataFrame({
        'feature': X_train.columns,
        'score': X_scored.scores_
    })
    head_feature_num = 18
    feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
    X_train_head = X_train[X_train.columns[X_train.columns.isin(feat_scored_headnum)]]
    X_scaled = pd.DataFrame(preprocessing.scale(X_train), columns=X_train.columns)
    X_test_head = X_test[X_test.columns[X_test.columns.isin(feat_scored_headnum)]]
    print("单变量选择后的shape")
    # pca_ = pca.PCA(n_components=0.99) #0.95
    # pca_.fit(X)
    # X = pd.DataFrame(pca_.transform(X))
    # print("PCA done.")
    print(X_train_head.shape)
    print(y_train.shape)
    print(X_test_head.shape)
    print("Start training......")
    svr, line, lasso, ENet, KRR1, KRR2, lgbm, xgb, nn = build_model()
    train_start = datetime.now()
    score = rmsle_cv(svr, X_train_head, y_train)
    print("SVR 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    svr.fit(X_train_head, y_train)
    score = rmsle_cv(line, X_train_head, y_train)
    print("Line 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(lasso, X_train_head, y_train)
    print("Lasso 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(ENet, X_train_head, y_train)
    print("ElasticNet 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(KRR2, X_train_head, y_train)
    print("Kernel Ridge2 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    KRR2.fit(X_train_head, y_train)
    # =============================================================================
    score = rmsle_cv(KRR1, X_train_head, y_train)
    print("Kernel Ridge1 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # =============================================================================
    head_feature_num = 22
    feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
    X_train_head3 = X_train[X_train.columns[X_train.columns.isin(feat_scored_headnum)]]
    X_scaled = pd.DataFrame(preprocessing.scale(X_train), columns=X_train.columns)
    score = rmsle_cv(xgb, X_train_head3, y_train)
    print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    xgb.fit(X_train_head, y_train)
    # =============================================================================
    head_feature_num = 22
    feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
    X_train_head4 = X_train[X_train.columns[X_train.columns.isin(feat_scored_headnum)]]
    X_scaled = pd.DataFrame(preprocessing.scale(X_train), columns=X_train.columns)
    score = rmsle_cv(lgbm, X_train_head4, y_train)
    print("LGBM 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    lgbm.fit(X_train_head, y_train)
    # =============================================================================
    head_feature_num = 18
    feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
    X_train_head5 = X_train[X_train.columns[X_train.columns.isin(feat_scored_headnum)]]
    X_scaled = pd.DataFrame(preprocessing.scale(X_train_head5), columns=X_train_head5.columns)
    score = rmsle_cv(nn, X_train_head5, y_train)
    print("NN 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    nn.fit(X_train_head, y_train)
    # =============================================================================
    averaged_models = AveragingModels(models=(svr, KRR2, lgbm, nn))
    score = rmsle_cv(averaged_models, X_train_head, y_train)
    print("对基模型集成后的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    averaged_models.fit(X_train_head, y_train)
    # stacking_models = StackingAveragedModels(base_models=(svr,KRR2,nn), meta_model=xgb)
    # stacking_models.fit(X_train_head.values, y_train.values)
    # stacked_train_pred = stacking_models.predict(X_train_head)
    # score = mean_squared_error(y_train.values, stacked_train_pred)
    # print("Stacking Averaged models predict score: {:.4f}".format(score))

    train_end = datetime.now()
    print('spend time:' + str((train_end - train_start).seconds) + '(s)')

    print("Predict......")
    # X_test = pd.DataFrame(pca_.transform(X_test))
    # X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    y_pred = averaged_models.predict(X_test_head)
    result = pd.DataFrame(y_pred)
    result.to_csv("predict.txt", index=False, header=False)
    print("Predict Done.")
    print(datetime.now())


main()
