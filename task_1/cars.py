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
from scipy import stats
from scipy.stats import norm, skew,kurtosis
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler  

cars=pd.read_csv('cars_price.csv')


#DATA CLEANING
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


#EDA
make_groupby=cars[['make', 'priceUSD']].groupby('make').max().sort_values(by='priceUSD', ascending=False)

mileage_groupby=cars[['year', 'priceUSD', 'mileage(kilometers)']].groupby('year').mean()
plt.plot(mileage_groupby) #blue line price

segment_groupby=cars[['segment', 'priceUSD']].groupby('segment').median().sort_values(by='priceUSD')
sns.displot(segment_groupby)

segment_groupby2=cars[['segment', 'priceUSD']].groupby('segment').agg(['mean', 'count'])
plt.plot(segment_groupby2) #orange line is count and vice versa

make_groupby=cars[['make', 'model', 'priceUSD']].groupby(['make', 'model']).agg(['mean', 'count']).sort_values(by=('priceUSD', 'mean'))

fuel_type_groupby=cars[['fuel_type', 'make', 'priceUSD']].groupby(['fuel_type', 'make']).max().sort_values(by='priceUSD', ascending=False)

fuel_type_groupby2=cars[['fuel_type', 'priceUSD']].groupby(['fuel_type']).count().sort_values(by='priceUSD')
plt.plot(fuel_type_groupby2)

fuel_type_groupby3=cars[['fuel_type', 'priceUSD']].groupby(['fuel_type']).median().sort_values(by='priceUSD')
plt.plot(fuel_type_groupby3)

transmission_type_groupby=cars[['transmission', 'drive_unit', 'priceUSD']].groupby(['transmission', 'drive_unit']).mean().sort_values(by='priceUSD')

color_groupby=cars[['color', 'segment', 'priceUSD']].groupby(['segment', 'color']).median().sort_values(by=['segment', 'priceUSD'], ascending=False)
color_groupby2=cars[['color', 'model', 'priceUSD']].groupby(['color', 'model']).median().sort_values(by=['priceUSD'], ascending=False)

#Visualizations
sns.boxplot(cars['priceUSD'])

#categorical visualization function
def cat_visualization(dataframe, col):
    sns.countplot(dataframe[col])
    return

cat_visualization(cars, 'segment')

cat_visualization(cars, 'fuel_type')

cat_visualization(cars, 'transmission')

cat_visualization(cars, 'color')

cat_visualization(cars, 'condition')

sns.countplot(y='drive_unit', data=cars)

#scatterplots
sns.scatterplot(x=cars['year'], y=cars['priceUSD'])

sns.scatterplot(x=cars['year'], y=cars['mileage(kilometers)'])

sns.scatterplot(x=cars['mileage(kilometers)'], y=cars['priceUSD'])
sns.scatterplot(x=cars['priceUSD'], y=cars['mileage(kilometers)'])

sns.scatterplot(x=cars['priceUSD'], y=cars['segment'])

"""
From the EDA we can infer the following from the data:
1. Segment D is the most popular segment followed by C
   The costliest segment type is the S type
2. Most popular fuel_type is petrol followed by diesel
   Electrocars are much costlier as compared to petrol and diesel
3. Mechanics is the preferred transmission type rather than auto
 . Auto transmission type is costlier than mechanical transmission. The costliest
   is auto with all wheel drive
4. In case of drive unit front wheel drive is the most popular
5. We can observe a trend of rise in prices every year
6. Even though with every passing year the price increases the mileage of the cars is decreasing
   The "Therma" by car maker Lancia is the cheapest in the dataset
   The "Bentayga" by car maker Bentley is the costliest followed by "Mulsanne" by Bentley as well
   The costlies car in the dataset is of purple color followed by white and grey.
    
"""

#Statistical analysis

sns.heatmap(cars.corr(), annot=True, robust=True)

#ho mileage and price are independent 
#h1 mileage and price are not independent
#significance level 0.05

a=cars['mileage(kilometers)']
b=cars['priceUSD']
print(stats.pearsonr(a, b))

#t test
stats.ttest_ind(a, b)

#since the p value is less than the significance level we reject the null hypothesis
#thus they are related

#encoding categorical variables
one_hot_list=['condition', 'fuel_type', 'transmission', 'drive_unit', 'segment', 'color']

cars2=cars.copy()

cars2=pd.get_dummies(cars2, columns=one_hot_list, drop_first=True)

le=LabelEncoder()
cars2['make']=le.fit_transform(cars2['make'])
cars2['model']=le.fit_transform(cars2['model'])

correlation=cars2.corr()

y=cars2['priceUSD']
X=cars2.drop('priceUSD', axis=1)

#splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#applying regression models
def reg_model(model):
    regressor=model
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    r2_s=r2_score(y_test, y_pred)
    mse=mean_squared_error(y_test, y_pred)
    rmse=np.sqrt(mse)
    print(model)
    print("r2 score:", r2_s)
    print("rmse score:", rmse)
    return

reg_model(DecisionTreeRegressor())
reg_model(RandomForestRegressor(n_estimators=100, random_state=42)) 
reg_model(LinearRegression())
reg_model(SVR(kernel = 'rbf')) 

# Random Forest gives the best R2 score and lowset RMSE followed by DT
    
