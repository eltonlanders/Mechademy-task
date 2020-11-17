# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:26:45 2020

@author: elton
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, chi2, f_regression

data=pd.read_csv('propulsion.csv')

data.drop('Unnamed: 0', axis=1, inplace=True)

#renaming columns
data.columns
data.columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate',
                     'gg_rate', 'sp_torque', 'pp_torque', 'hpt_temp',
                     'gt_c_i_temp', 'gt_c_o_temp', 'hpt_pressure',
                     'gt_c_i_pressure', 'gt_c_o_pressure',
                     'gt_exhaust_pressure', 'turbine_inj_control', 'fuel_flow',
                     'gt_c_decay',  'gt_t_decay']

#checking for misssing values
data.isnull().sum()

#dropping two constant value columns GT compressor inlet air temperature and 
#GT compressor inlet air pressure
data.drop(['gt_c_i_temp', 'gt_c_i_pressure'], axis=1 ,inplace=True)

data.info()

data.describe()

pearson_correlation=data.corr('pearson')
spearman_correlation=data.corr('spearman')

#visualizations
data['lever_position'].iloc[0:50].plot()

data['ship_speed'].iloc[0:50].plot()

data['gt_shaft'].iloc[0:50].plot()

sns.pairplot(data.iloc[:, [2, 3, 4, 14]], hue='gt_c_decay')

sns.pairplot(data.iloc[:, [5, 6, 7, 14]], hue='gt_c_decay')

sns.pairplot(data.iloc[:, [8, 9, 10, 14]], hue='gt_c_decay')

sns.pairplot(data.iloc[:, [11, 12, 13, 14]], hue='gt_c_decay')

sns.pairplot(data.iloc[:, [2, 3, 4, 15]], hue='gt_t_decay')

sns.pairplot(data.iloc[:, [5, 6, 7, 15]], hue='gt_t_decay')

sns.pairplot(data.iloc[:, [8, 9, 10, 15]], hue='gt_t_decay')

sns.pairplot(data.iloc[:, [11, 12, 13, 14, 15]], hue='gt_t_decay')

#splitting
data2=data.values
X = data2[:, 0:14]
y= data2[:, 14] #GT Compressor decay state coefficient
y2=data2[:, 15] #GT Turbine decay state coefficient

# feature selection for GT compressor decay state coefficient
X_train, X_test, y_train1, y_test1 = train_test_split(X, y, test_size=0.20, random_state=0)

f_selector = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
f_selector.fit(X_train, y_train1)
# transform train input data
X_train_fs = f_selector.transform(X_train)
# transform test input data
X_test_fs = f_selector.transform(X_test)
# Plot the scores for the features
plt.bar([i for i in range(len(f_selector.scores_))], f_selector.scores_)
plt.xlabel("feature index")
plt.ylabel("F-value (transformed from the correlation values)")
plt.show()
# The top selected features for GT compressor decay state coefficient are 7 8 12 11 and 4 


# feature selection for GT Turbine decay state coefficient
X_train, X_test, y_train2, y_test2 = train_test_split(X, y2, test_size=0.20, random_state=0)

f_selector = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
f_selector.fit(X_train, y_train2)
# transform train input data
X_train_fs = f_selector.transform(X_train)
# transform test input data
X_test_fs = f_selector.transform(X_test)
# Plot the scores for the features
plt.bar([i for i in range(len(f_selector.scores_))], f_selector.scores_)
plt.xlabel("feature index")
plt.ylabel("F-value (transformed from the correlation values)")
plt.show()
#The top selected features for GT Turbine decay state coefficient are 7 11 10 12 and 4

# models for GT Compressor decay state coefficient
def reg_model(model):
    regressor=model
    regressor.fit(X_train, y_train1)
    y_pred=regressor.predict(X_test)
    r2_s=r2_score(y_test1, y_pred)
    mse=mean_squared_error(y_test1, y_pred)
    rmse=np.sqrt(mse)
    print(model)
    print("r2 score:", r2_s)
    print("rmse score:", rmse)
    return

reg_model(RandomForestRegressor(n_estimators=100, random_state=42)) 
reg_model(DecisionTreeRegressor())
reg_model(LinearRegression())
reg_model(SVR(kernel = 'rbf')) 

#models for GT Turbine decay state coefficient
def reg_model(model):
    regressor=model
    regressor.fit(X_train, y_train2)
    y_pred=regressor.predict(X_test)
    r2_s=r2_score(y_test2, y_pred)
    mse=mean_squared_error(y_test2, y_pred)
    rmse=np.sqrt(mse)
    print(model)
    print("r2 score:", r2_s)
    print("rmse score:", rmse)
    return

reg_model(RandomForestRegressor(n_estimators=500, random_state=42)) 
reg_model(DecisionTreeRegressor())
reg_model(LinearRegression())
reg_model(SVR(kernel = 'rbf'))

#creating an ANN for GT Compressor decay state coefficient
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
ann.fit(X_train, y_train1, batch_size = 32, epochs = 100)
y_pred = ann.predict(X_test)

gtcc_ann_mse = mean_squared_error(y_test1, y_pred)
gtcc_ann_rmse = np.sqrt(gtcc_ann_mse)
gtcc_ann_mse = (y_test1, y_pred)
gtcc_ann_r2_score = r2_score(y_test1, y_pred)
print('GTCC mean square error:', gtcc_ann_mse)
print('GTCC root mean square error:', gtcc_ann_rmse)
print('GTCC R2 score:', gtcc_ann_r2_score)

#creating an ANN for predicting GT Turbine decay state coefficient
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
ann.fit(X_train, y_train2, batch_size = 32, epochs = 100)
y_pred = ann.predict(X_test)

gttc_ann_mse = mean_squared_error(y_test2, y_pred)
gttc_ann_rmse = np.sqrt(gttc_ann_mse)
gttc_ann_mse = (y_test2, y_pred)
gttc_ann_r2_score = r2_score(y_test2, y_pred)
print('GTTC mean square error:', gttc_ann_mse)
print('GTTC root mean square error:', gttc_ann_rmse)
print('GTTC R2 score:', gttc_ann_r2_score)