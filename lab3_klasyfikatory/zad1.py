from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from statistics import mode
import numpy as np

iris = datasets.load_iris()

# Podział zbióru na uczący i testowy
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.7)

k = 5  # parametr k

# Utworzenie tablic
dst = []
predictions = []
neighbors = []

for i in range(features_test.shape[0]):  # Petla po elementach ze zbioru testowego
    for j in range(features_train.shape[0]):  # Pętla po elementach ze zbioru uczącego
        # Wyznaczenie odegosci i przypisanie wraz z y_test
        dst.append([distance.euclidean(features_test[i, 0:4], features_train[j, 0:4]), labels_train[j]])
        dst.sort()  # Posortowanie po odlegosci

    for x in range(k):  # Petla po somsiadach
        neighbors.append(dst[x][1])  # Wybieramy k najblizszych

    predictions.append(mode(neighbors))  # mode wybiera najczestrzy element
    dst.clear()
    neighbors.clear()

# Sprawdzanie skuteczności klasyfikatora
output = accuracy_score(labels_test, predictions)
print("wyni ", output)
