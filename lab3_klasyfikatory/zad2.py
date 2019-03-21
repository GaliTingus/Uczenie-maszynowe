# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:38:51 2019

@author: student189
"""

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
 
#Podziel zbiór na uczący i testowy, test_size - procentowy udział (przykład 50 % uczący i testowy)
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.5)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(features_train, labels_train)
predictions = neigh.predict(features_test)

# Sprawdzanie skuteczności klasyfikatora
output = accuracy_score(labels_test, predictions)
print(output)