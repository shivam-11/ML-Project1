# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:40:25 2019

@author: Shivam Singh
"""

'''HOUSE_DATA'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump,load


house_data = pd.read_csv("house.csv")
print(house_data.head())                       #giving top five value
print(house_data.info())                       #information about house_data
print(house_data['CHAS'].value_counts() )      #information about CHAS feature 
print(house_data.describe())                   #complete description about house_data

'''Ploting histogram'''
house_data.hist(bins=50,figsize=(20,15))
plt.show()

'''train_test_split'''

train_data, test_data = train_test_split(house_data,test_size=0.2,random_state=42)
print("Rows in train set:",len(train_data))   # giving information about length train data
print("Rows in test set:",len(test_data))
print(test_data['CHAS'].value_counts())    # giving information about length test data


'''straitifiedshufflesplit:it is used to split categorical data in equal proprtion'''
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state = 42)  #n_split: numberof time to resuffle
for train_index,test_index in split.split(house_data,house_data['CHAS']):
    strat_train = house_data.loc[train_index]  #loc is used to access all row and column
    strat_test = house_data.loc[test_index]
print(strat_train['CHAS'].value_counts())   

house_data = strat_train.copy() # working in imputer
''' Correlation Matrix ''' 

corr_matrix = house_data.corr()
print(corr_matrix['MEDV'].sort_values(ascending = False))

'''Scatter plot:this is used to plot with whose are highly correlated'''
attr = ['RM','ZN','PTRATIO','LSTAT']
scatter_matrix(house_data[attr],figsize = (10,10))
house_data.plot(kind="scatter",x="RM",y="MEDV",alpha=0.7)

'''Attribute Combination: Creating new attribute'''
house_data['TPR'] = house_data['TAX']/house_data['RM']
corr_matrix = house_data.corr()
print(corr_matrix['MEDV'].sort_values(ascending = False))
house_data.plot(kind="scatter",x="TPR",y="MEDV",alpha=0.7)


house_data = strat_train.drop("MEDV",axis=1)
house_label = strat_train["MEDV"].copy()
''' Missing Values in Atttribute 
    1:Remove all the attribute value of that row
    2:Remove that column whose value is missing if its correlation is very less
    3:Place mean value at the missing position
'''
#1
a= house_data.dropna(subset = ["RM"])
print(a.shape)

#2
b=house_data.drop("RM",axis=1)
print(b)
print(b.shape)

#3
median = house_data["RM"].median()
print(median)
print(house_data["RM"].fillna(median))

imputer = SimpleImputer(strategy = "median")
imputer.fit(house_data)
print(imputer.statistics_)
x=imputer.transform(house_data)
housing_tr = pd.DataFrame(x,columns=house_data.columns)
print(housing_tr.describe())

'''Pipeline'''
my_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = "median")),
        ('std_scalar', StandardScaler()),
        ])
 
housing_num_tr = my_pipeline.fit_transform(house_data) 
print(housing_num_tr)

'''Model:LinearRegression'''

#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr,house_label)

house_pred = model.predict(housing_num_tr)
lin_mse = mean_squared_error(house_label,house_pred)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)
scores = cross_val_score(model,housing_num_tr,house_label, scoring="neg_mean_squared_error",cv=10 )
rmse =np.sqrt(-scores)
 
print("scores:",rmse)
print("mean:",rmse.mean())
print("standard deviation:",rmse.std())
some_data = house_data.iloc[:5]
some_label= house_label.iloc[:5]
prepared_data =  my_pipeline.transform(some_data)
pred_data=model.predict(prepared_data)
print(pred_data)
print(list(some_label))

dump(model,'realstate.joblib')














