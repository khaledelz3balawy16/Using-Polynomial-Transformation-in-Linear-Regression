# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:45:04 2024

@author: khaled elz3blawy
"""
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('satf.csv')
print("First 10 rows of the dataset:")
print(dataset.head(10))
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

print("X values:")
print(X)
print("y values:")
print(y)
## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size=0.2 , random_state=0)
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
print("y_train:")
print(y_train)
print("y_test:")
print(y_test)
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.transform(X_test)


print("Shape of X_train_poly:")
print(X_train_poly.shape)
print("Shape of X_test_poly:")
print(X_test_poly.shape)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

y_pred = lin_reg.predict(X_test_poly)

print("Predicted y values:")
print(y_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

print("Mean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:")
print(mean_squared_error(y_test, y_pred))
print("Median Absolute Error:")
print(median_absolute_error(y_test, y_pred))