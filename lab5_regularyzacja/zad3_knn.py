from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# Podziel zbiór na uczący i testowy, test_size - procentowy udział (przykład 50 % uczący i testowy)
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.5)

k = np.linspace(1, 15, 15, dtype=int)
accuracy_train = []
accuracy_test = []
k_opt = 0
val_opt = 0
for k_value in k:
    neigh = KNeighborsClassifier(n_neighbors=k_value)
    neigh.fit(features_train, labels_train)
    predictions = neigh.predict(features_test)

    # Sprawdzanie skuteczności klasyfikatora
    output = accuracy_score(labels_test, predictions)
    accuracy_test.append(output)
    if output > val_opt:
        val_opt = output
        k_opt = k_value
# print(output)
plt.figure()
plt.plot(k, accuracy_test)
plt.title("Dokładność zbioru testowego KNN")
plt.xlabel('k')
plt.ylabel('Dokładność')
plt.show()

print("\nK optymalne = ", k_opt)
print("Dokładność dopasowania opymalnego = ", val_opt)
