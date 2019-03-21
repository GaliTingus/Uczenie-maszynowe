from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
 
#Podziel zbiór na uczący i testowy, test_size - procentowy udział (przykład 50 % uczący i testowy)
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.5)
print(features_train)
print(labels_train)
#Przykład użycia odległości euklidesowej
print(features_test.shape)
k=5
dstClosest = np.array(np.zeros([features_test.shape[0], 3, k+1]))
#dst = np.array(np.zeros([features_test.shape[0], 1]), dtype=[('Distance', 'float64'), ('index', int)])
dst = []

#   count distances
for i in range(features_test.shape[0]):
    for j in range(features_train.shape[0]):
        dst.append([distance.euclidean(features_test[i,0:4], features_train[j,0:4]), j, labels_train[i]])
        dst.sort()
    for number in range(k):
        dstClosest[i, 0, number] = dst[number+1][0]
        dstClosest[i, 1, number] = dst[number+1][1]
        dstClosest[i, 2, number] = dst[number+1][2]
    dst = []
        
predictions = np.median(dstClosest[:,2,:], axis=1)
print(predictions)

# Sprawdzanie skuteczności klasyfikatora
output = accuracy_score(labels_test, predictions)
print(output)