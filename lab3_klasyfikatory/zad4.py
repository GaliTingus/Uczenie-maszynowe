# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


def plot_mnist(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.05)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        # plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# wczytywanie danych
dane = loadmat('baza_mnist.mat')

# Zad 1. Podziel dane na parametry X oraz odpowiedź y:
X = dane['X']
y = dane['y']

# Standaryzacja
for i in range(X.shape[0]):
    X[i, :] = X[i, :] / np.std(X[i, :])

# Zamiana cyfry 10 -> 0 (błąd w zbiorze danych)
y[np.where(y == 10)] = 0

# wysokość i szerokość obrazka z cyfrą
h = 20
w = 20

plot_mnist(X[0:5000:200, :], y[0:5000:200], h, w)
plt.show()
#
# # Zad 2. Proszę wyświetlić liczbę cyfr oraz liczbę pikseli przypadającą na jeden obraz
tmp, count = np.unique(y, return_counts=True)
print("Ilość obrazków = ", len(tmp))
print("Pikseli na cyfrę = ", count[0] * h * w)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

DEPTH = 10
MAX_FEAT = 100
clf = tree.DecisionTreeClassifier()  # instancja klasyfikatora DecisionTreeClassifier
clf.fit(X_train, y_train)  # fitowanie do danych treningowych
y_pred = clf.predict(X_test)  # predykcja na danych testowych

# Pokaż kilka przykładowych klasyfikacji:
plot_mnist(X_test[0:40, :], y_pred[0:40], h, w, n_row=5, n_col=8)
plt.show()

# uzupełnij miary klasyfikacji
f1 = f1_score(y_test, y_pred, average='micro')
print("wynik F1: ", f1)  # uzupełnij
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("raport klasyfikacji:")
print(classification_report(y_test, y_pred))
