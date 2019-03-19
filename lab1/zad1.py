# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:14:06 2019

@author: student189
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris

# load the famous iris data
irisRaw = load_iris()

# Konwersja danych do pakietu Panda
# read iris.data into a pandas DataFrame (df), including column names
iris = pd.DataFrame(data=np.c_[irisRaw['data'], irisRaw['target']],
                    columns=irisRaw['feature_names'] + ['target'])

print(iris)
print("------------------------------------------------------")

print("kolumny: ", iris.shape[1])
print("wiersze: ", iris.shape[0])
print("------------------------------------------------------")
print("opis")
discrb = iris.describe()
print(discrb)

print("Grupowanie")
print("------------------------------------------------------")
print(iris.groupby('target').count())

print("------------------------------------------------------")
print("Piersze 5 elementów")
print(iris.head(5))

print("------------------------------------------------------")
print("Brakuje danych?", iris.shape[0] - iris.dropna(how='any').shape[0])

print("------------------------------------------------------")
print("Sortowanie po 2 kolumnie")
iris2 = iris.sort_index(by='sepal width (cm)')
print(iris2)

print("------------------------------------------------------")
print("Min = " + str(iris[iris.columns[2]].min()) + " o adresie = " + str(iris[iris.columns[2]].idxmin()))
print("Max = " + str(iris[iris.columns[2]].max()) + " o adresie = " + str(iris[iris.columns[2]].idxmax()))

print("------------------------------------------------------")
print("Odchylenie:")
for i in iris.columns:
    print("Kolumna", i, "\t\todchylenie = " + str(iris[i].std()))

print("------------------------------------------------------")
plt.hist(iris[iris.columns[0]], 40)
plt.hist(iris[iris.columns[1]], 40)
plt.hist(iris[iris.columns[2]], 40)
plt.hist(iris[iris.columns[3]], 40)
plt.legend(iris.columns)
plt.show()