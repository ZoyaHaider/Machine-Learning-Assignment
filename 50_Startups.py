# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:54:15 2020

@author: ZOYA

"""
#PLYNOMIAL REGRESSION

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#CALIFORNIA

# Importing the dataset
dataset = pd.read_csv('50_startups.csv')
A=dataset.loc[dataset['State'] == 'California','R&D Spend']
B=dataset.loc[dataset['State'] == 'California','Administration']
C=dataset.loc[dataset['State'] == 'California','Marketing Spend']
D=A+B+C
E=dataset.loc[dataset['State'] == 'California','Profit']
D=np.arange(17).reshape(-1,1)
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
D_train, D_test, E_train, E_test = train_test_split(D, E, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
lin_reg.fit(D, E)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
D_poly = poly_reg.fit_transform(D)
poly_reg.fit(D_poly, E)
lin_reg= LinearRegression()
lin_reg.fit(D_poly, E)

# Visualising the Polynomial Regression results
plt.scatter(D, E, color = 'maroon')
plt.plot(D, lin_reg.predict(poly_reg.fit_transform(D)), color = 'grey')
plt.title('CALIFORNIA')
plt.xlabel('STARTUP')
plt.ylabel('PROFIT')
plt.show()

E_pred=lin_reg.predict(poly_reg.fit_transform([[20]]))
print('Profit of California is\n',E_pred)


#NEW YORK
F=dataset.loc[dataset['State'] == 'New York','R&D Spend']
G=dataset.loc[dataset['State'] == 'New York','Administration']
H=dataset.loc[dataset['State'] == 'New York','Marketing Spend']
I=F+G+H
J=dataset.loc[dataset['State'] == 'New York','Profit']
I=np.arange(17).reshape(-1,1)
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
I_train, I_test, J_train, J_test = train_test_split(I, J, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(I, J)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
I_poly = poly_reg.fit_transform(I)
poly_reg.fit(I_poly, J)
lin_reg_2= LinearRegression()
lin_reg_2.fit(I_poly, J)

# Visualising the Polynomial Regression results
plt.scatter(I, J, color = 'maroon')
plt.plot(I, lin_reg_2.predict(poly_reg.fit_transform(I)), color = 'GREY')
plt.title('NEW YORK')
plt.xlabel('STARTUP')
plt.ylabel('PROFIT')
plt.show()

J_pred=lin_reg_2.predict(poly_reg.fit_transform([[20]]))
print('Profit of New York is\n',J_pred)
print('\n\t\'NEW YORK WILL PROVIDE BEST PROFIT IN FUTURE\'')