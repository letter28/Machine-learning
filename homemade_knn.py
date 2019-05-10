# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:42:51 2019

@author: zero-
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
style.use('fivethirtyeight')

ds = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

def knn(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less then total voting groups!')
    distances = []
    for group in data:
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

#Preparing the dataset for knn function
data = pd.read_csv('breast-cancer-wisconsin.data')
data.replace('?', np.nan, inplace=True)
data.dropna(0, inplace=True)
data.drop('id', 1, inplace=True)
full_data = data.astype(float).values.tolist()

test_size = 0.3
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for patient in test_set[group]:
        vote = knn(train_set, patient, k=5)
        if vote == group:
            correct += 1
        total += 1

print('Accuracy: ', correct/total)

#for i in ds:
#    for ii in ds[i]:
#        plt.scatter(ii[0], ii[1], color=i, s=100)
#plt.show()



#new = knn(data, [4,8], k=3)
#print(new)