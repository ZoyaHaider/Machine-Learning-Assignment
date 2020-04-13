# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:27:52 2020

@author: zoya

"""
#Data Of Monthly Experience And Income

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('monthlyexp vs incom.csv')
A = dataset.iloc[:, 0:1].values
B = dataset.iloc[:, 1].values

# Splittin the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(A_train, B_train)

# Predicting the Test set results
B_pred = regressor.predict(A_test)

# Visualising the Training set results
plt.scatter(A_train, B_train, color = 'PINK')
plt.plot(A_train, regressor.predict(A_train), color = 'BLACK')
plt.title('Incom vs Experience (Training set)')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.show()

# Visualising the Test set results
plt.scatter(A_test, B_test, color = 'PINK')
plt.plot(A_train, regressor.predict(A_train), color = 'BLACK')
plt.title('Incom vs Experience (Test set)')
plt.xlabel('Monthly Expirience')
plt.ylabel('Incom')
plt.show()
