import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os


def sign(t):
    return 1.0 / (1.0 + np.exp(-t))


def h(X, theta):
    return sign(np.matrix(theta).transpose() * np.matrix(X))


def cost(X, y, theta):
    m = X.shape[1]
    first = - np.multiply(y, np.log(h(X, theta)))
    secend = np.multiply((1.0 - y), np.log(1.0 - h(X, theta)))
    return np.sum(first - secend) / m


def gradient_prosty(X, y, theta, alpha, it):
    m = X.shape[1]
    for i in range(it):
        j0 = np.sum(h(X, theta) - y) / m
        j1 = np.sum(np.multiply((h(X, theta) - y), X[1])) / m
        j2 = np.sum(np.multiply((h(X, theta) - y), X[2])) / m
        theta[0] = theta[0] - (alpha * j0)
        theta[1] = theta[1] - (alpha * j1)
        theta[2] = theta[2] - (alpha * j2)
        comp_cost = cost(X, y, theta)
    return theta, comp_cost


path = os.getcwd() + '/dane_egz.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])

data_std = (data - data.mean()) / (data.std())

# print("Describe:\n", data.describe())
# print("Head:\n", data.head(20))

# Wyswietlenie danych
X = np.array(data.values[:, :2])
y = np.array(data.values[:, 2])

plt.plot(X[data.values[:, 2] == 1, 0], X[data.values[:, 2] == 1, 1], 'ro', label='Przyjęty')
plt.plot(X[data.values[:, 2] == 0, 0], X[data.values[:, 2] == 0, 1], 'bo', label='Odrzucony')
plt.legend()
plt.show()

# Funkcja kosztu
ones = np.ones([data[data.columns[0]].shape[0]])

X = np.matrix([ones, data_std.Exam1, data_std.Exam2])
y = np.array([data.Admitted])
theta = np.zeros([X.shape[0], 1])

print("Koszt = ", cost(X, y, theta))

# gradient
alpha = 1
it = 150

theta, comp_cost = gradient_prosty(X, y, theta, alpha, it)

print("Theta = ", theta)
print("Koszt = ", comp_cost)

# skutecznosc
y_pred = h(X, theta)
y_pred[np.where(y_pred > 0.5)] = 1
y_pred[np.where(y_pred <= 0.5)] = 0

print('Efektywność: ', sklearn.metrics.accuracy_score(data['Admitted'], np.array(y_pred).reshape(100)))
