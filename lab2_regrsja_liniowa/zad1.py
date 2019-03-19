import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def computeCost(X, y, theta):
    x = np.ones((X.shape[0] + 1, X.shape[1]))
    x[-1:, :] = X
    kwadraty = np.power((theta * x) - y, 2)
    return (np.sum(kwadraty)) / (2 * X.shape[1])


path = os.getcwd() + '/dane1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

print(data.head())
print(data.describe())

plt.plot(data['Population'], data['Profit'], 'ro')
# plt.show()

X = data['Population']
y = data['Profit']
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0.0, 0.0]))
wynik = computeCost(X, y, theta)
print(wynik)


def theta_x(X, theta):
    x = np.ones((X.shape[0] + 1, X.shape[1]))
    x[-1:, :] = X
    return theta * x


def gradient_simple(X, y, theta, alpha, it, cost):
    for i in range(it):
        j0 = np.sum(theta_x(X, theta) - y) / (X.shape[1])
        j1 = (((theta_x(X, theta) - y) / (X.shape[1])) * np.transpose(X))

        theta[0, 0] = theta[0, 0] - (alpha * j0)
        theta[0, 1] = theta[0, 1] - (alpha * j1)
        cost[i] = computeCost(X, y, theta)
    return theta, cost



alpha = 0.01
it = 1000
# theta = np.array([0, 0])
cost = np.zeros(it, float)
theta, cost = gradient_simple(X, y, theta, alpha, it, cost)
# print(countHypotesis(X, theta))
print("________________________________________")
print("Theta: " + str(theta))
print("Cost: " + str(cost[it - 1]))

x = np.linspace(5, 23, 2)
y = theta[0, 1] * x + theta[0, 0]
plt.plot(x, y)
plt.show()
plt.figure()

plt.plot(cost)
plt.xlabel("iteracje")
plt.ylabel("funkcja kosztu")
plt.show()
