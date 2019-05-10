# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:09:08 2019

@author: zero-
"""

import numpy as np
import pandas as pd
from sklearn import model_selection, svm

pd.options.display.max_columns = 20
pd.options.display.max_rows = 200
pd.options.display.width = 500

data = pd.read_csv('breast-cancer-wisconsin.data')
data.replace('?', np.nan, inplace=True)
data.dropna(0, inplace=True)
data.drop('id', 1, inplace=True)

#Features
X = np.array(data.drop(['class'], 1), dtype=np.int64)
#Class that we're trying to predict
Y = np.array(data['class'])

#Splitting the data into training and testing datasets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
#Declaring a classifier and fitting it to our training data
clf = svm.SVR()
clf2 = svm.SVC()
clf.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
accuracy2 = clf2.score(X_test, Y_test)
print('Accuracy for SVM SVR: ', accuracy)
print('Accuracy for SVM SVC: ', accuracy2)