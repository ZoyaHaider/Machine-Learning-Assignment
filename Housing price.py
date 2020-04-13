# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 03:33:10 2020

@author: ZOYA
"""

# Housing price according to the ID is assigned to every-house. Perform future analysis
# where when ID is inserted the housing price is displayed.

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Housing Price')
plt.xlabel('ID')
plt.ylabel('House Price')
plt.show()



# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Housing Price')
plt.xlabel('ID')
plt.ylabel('House Price')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Housing Price')
plt.xlabel('ID')
plt.ylabel('House Price')
plt.show()

print('Predicting a new result with Linear Regression')

A=lin_reg.predict([[2921]])
B=lin_reg.predict([[2921]])
C=lin_reg.predict([[2922]])
print('The housing price foe ID 2920 is',A)
print('The housing price foe ID 2920 is',B)
print('The housing price foe ID 2920 is',C)

print('\n\n Predicting a new result with Polynomial Regression')
D=lin_reg_2.predict(poly_reg.fit_transform([[2920]]))
E=lin_reg_2.predict(poly_reg.fit_transform([[2921]]))
F=lin_reg_2.predict(poly_reg.fit_transform([[2922]]))

print('The housing price foe ID 2920 is',D)
print('The housing price foe ID 2920 is',E)
print('The housing price foe ID 2920 is',F)
