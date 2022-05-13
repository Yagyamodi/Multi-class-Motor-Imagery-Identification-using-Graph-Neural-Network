# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:00:29 2021

@author: 91876
"""

# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import json
from utilities import process

def process1(path):
    data = json.load(open(path))
    return data

"""
# loading the iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target
print(X.shape, "\n", y.shape)
"""

path = "C:/Users/91876/Downloads/SimGNN-main/dataset/A06/Training/A06T_"
X_train = []
for i in range(1,5):
    data = np.array(process(path + str(i) + "/")["adj_matrix_1"])
    np.append(X_train, data)

path = "C:/Users/91876/Downloads/SimGNN-main/dataset/A06/Testing/A06E_"

X_test = np.array([])
for i in range(1,5):
    data = np.array(process(path + str(i) + "/")["adj_matrix_1"])
    np.append(X_test, data)
    
y_test = np.full((225,), 1)
for i in range(2,5):
    np.append(y_test, np.full((225,),i))
y_train = y_test

# dividing X, y into train and test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
print(cm)