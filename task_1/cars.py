# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:07:19 2020

@author: elton
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler

cars=pd.read_csv('cars_price.csv')

#dropping the 1st col as its redundant
cars.drop('Unnamed: 0', axis=1, inplace=True)

cols=cars.columns

#imputing null values of numerical columns by their mean
values=2103.201676 
cars['volume(cm3)'].fillna(value=values, inplace=True)

#imputing categorical null values with mode
cars = cars.fillna(cars.mode().iloc[0])

cars.isnull().sum()

head = cars.head(20)

#Visualizations
sns.boxplot(cars['priceUSD'])

#categorical
def cat_visualization(dataframe, col):
    sns.countplot(dataframe[col])
    return

cat_visualization(cars, 'segment')

cat_visualization(cars, 'fuel_type')

cat_visualization(cars, 'transmission')

cat_visualization(cars, 'condition')

cat_visualization(cars, 'drive_unit')

#scatterplots
sns.scatterplot(x=cars['year'], y=cars['priceUSD'])

sns.scatterplot(x=cars['year'], y=cars['mileage(kilometers)'])

sns.scatterplot(x=cars['mileage(kilometers)'], y=cars['priceUSD'])
sns.scatterplot(x=cars['priceUSD'], y=cars['mileage(kilometers)'])

sns.scatterplot(x=cars['priceUSD'], y=cars['segment'])
sns.scatterplot(x=cars['segment'], y=cars['priceUSD'])

make_groupby=cars[['make', 'priceUSD']].groupby('make').count()

mileage_groupby=cars[['year', 'priceUSD', 'mileage(kilometers)']].groupby('year').count()

segment_groupby=cars[['segment', 'priceUSD']].groupby('segment').count()
sns.displot(segment_groupby)

segment_groupby=cars[['segment', 'priceUSD']].groupby('segment').agg(['mean', 'count'])
plt.plot(segment_groupby)

segment_groupby=cars[['segment', 'priceUSD']].groupby('segment').agg(['mean', 'count'])
plt.plot(segment_groupby) #orange line is count and vice versa

make_groupby=cars[['make', 'model', 'priceUSD']].groupby(['make', 'model']).agg(['mean', 'count']).sort_values(by='model')

fuel_type_groupby=cars[['fuel_type', 'make', 'priceUSD']].groupby(['fuel_type', 'make']).max().sort_values(by='priceUSD', ascending=False)

fuel_type_groupby2=cars[['fuel_type', 'priceUSD']].groupby(['fuel_type']).count()
plt.plot(fuel_type_groupby2)

fuel_type_groupby2=cars[['fuel_type', 'priceUSD']].groupby(['fuel_type']).median()
plt.plot(fuel_type_groupby2)

fuel_type_groupby3=cars[['fuel_type', 'transmission', 'drive_unit', 'priceUSD']].groupby(['fuel_type', 'transmission', 'drive_unit']).mean().sort_values(by='priceUSD')

color_groupby=cars[['color','segment', 'priceUSD']].groupby(['segment', 'color']).median().sort_values(by=['segment', 'priceUSD'], ascending=False)

one_hot_list=['condition', 'fuel_type', 'transmission', 'drive_unit', 'segment', 'color']

cars2=cars.copy()

cars2=pd.get_dummies(cars2, columns=one_hot_list, drop_first=True)

le=LabelEncoder()
cars2['make']=le.fit_transform(cars2['make'])
cars2['model']=le.fit_transform(cars2['model'])

correlation=cars2.corr()

y=cars2['priceUSD']
X=cars2.drop('priceUSD', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
print(mse)
rmse=np.sqrt(mse)
print(rmse)
mean_absolute_error(y_test, y_pred)
r2_score(y_test, y_pred)


