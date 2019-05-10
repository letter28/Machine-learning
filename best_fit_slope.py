# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:36:50 2019

@author: zero-
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data = pd.read_csv('D:/Python/ML/Crobarometar/stranke_zz.csv')
#
#X = data['dana']
#Y = data['zz']

X = np.array([1,2,3,4,5,6,7,8,9], dtype=np.float64)
Y = np.array([4,5,6,4,7,8,6,9,12], dtype=np.float64)

def best_fit_line_and_intercept(X,Y):
    m = ((mean(X)*mean(Y)) - mean(X*Y)) / ((mean(X)*mean(X)) - mean(X**2))
    b = mean(Y) - mean(X)*m
    return m, b

def squared_err(Y_orig, Y_line):
    return sum((Y_line-Y_orig)**2)

def coef_of_determination(Y_orig, Y_line):
    Y_mean_line = [mean(Y_orig) for y in Y_orig]
    sq_error_regr = squared_err(Y_orig, Y_line)
    sq_error_y_mean = squared_err(Y_orig, Y_mean_line)
    return 1 - (sq_error_regr/sq_error_y_mean)


m,b = best_fit_line_and_intercept(X,Y)
print(m,b)

r_line = [(m*x)+b for x in X]
r_squared = coef_of_determination(Y, r_line)
print('R squared value for our regression line is: ', r_squared)
plt.plot(X,Y)
plt.plot(X,r_line)
plt.show()