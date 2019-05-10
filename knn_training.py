# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:09:08 2019

@author: zero-
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors, model_selection

pd.options.display.max_columns = 20
pd.options.display.max_rows = 200
pd.options.display.width = 500

data = pd.read_csv('breast-cancer-wisconsin.data')
data.replace('?', np.nan, inplace=True)
data.dropna(0, inplace=True)
data.drop('id', 1, inplace=True)
print(data.info())
#plt.matshow(data.corr())
#plt.xticks(range(len(data.columns)), data.columns, rotation=45)
#plt.yticks(range(len(data.columns)), data.columns)
#plt.colorbar()
#plt.show()

#Features
X = np.array(data.drop(['class'], 1), dtype=np.int64)
#Class that we're trying to predict
Y = np.array(data['class'])

#Splitting the data into training and testing datasets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.4)
#Declaring a classifier and fitting it to our training data
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)