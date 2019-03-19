import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

path = os.getcwd() + '/dane2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()

print(data2.describe())
data2[data2.columns[0]] = (data2[data2.columns[0]] - data2[data2.columns[0]].mean()) / data2[data2.columns[0]].std()
data2[data2.columns[1]] = (data2[data2.columns[1]] - data2[data2.columns[1]].mean()) / data2[data2.columns[1]].std()
data2[data2.columns[2]] = (data2[data2.columns[2]] - data2[data2.columns[2]].mean()) / data2[data2.columns[2]].std()
print(data2.describe())


def computeCost(X, y, theta):
    x = np.ones((X.shape[0] + 1, X.shape[1]))
    for i in range(1, X.shape[0] + 1):
        x[i, :] = X[i - 1]
    kwadraty = np.power((theta * x) - y, 2)
    return (np.sum(kwadraty)) / (2 * X.shape[1])



X = data2['Size']
X2 = data2['Bedrooms']
y = data2['Price']
X = np.matrix([X.values, X2.values])
y = np.matrix(y.values)
theta = np.matrix(np.array(np.zeros(X.shape[0] + 1,float)))
wynik = computeCost(X, y, theta)
print(wynik)


def theta_x(X, theta):
    x = np.ones((X.shape[0] + 1, X.shape[1]))
    for i in range(1, X.shape[0] + 1):
        x[i, :] = X[i - 1]
    return theta * x


def gradient_simple(X, y, theta, alpha, it, cost):
    for i2 in range(it):
        j = np.matrix(np.array(np.ones(X.shape[0]+1)))
        j[0,0] = np.sum(theta_x(X, theta) - y) / (X.shape[1])
        for i in range(1, X.shape[0]+1):
            j[0,i] = (((theta_x(X, theta) - y) / (X.shape[1])) * np.transpose(X[i-1]))
        theta = theta - alpha*j
        cost[i2] = computeCost(X, y, theta)
    return theta


alpha = 0.01
it = 1000

cost = np.zeros(it, float)
theta = gradient_simple(X, y, theta, alpha, it, cost)
print("________________________________________")
print("Theta: " + str(theta))
print("Cost: ", cost[it - 1])


Y_pred = np.zeros(X.shape[1])
Y_pred = theta[0,0] + X[0]*theta[0,1]+ X[1]*theta[0,2]
fig = plt.figure()
Axes3D = fig.add_subplot(111, projection='3d')
Axes3D.scatter(X[0], X[1], y, zdir='z', s=20, c='r', depthshade=True)
Axes3D.scatter(X[0], X[1], Y_pred, zdir='z', s=20, c='b', depthshade=True)
Axes3D.set_xlabel('Wielkość mieszkania')
Axes3D.set_ylabel('Ilosć sypialni')
Axes3D.set_zlabel('Koszt')
plt.show()

plt.figure()
plt.plot(cost)
plt.xlabel("iteracje")
plt.ylabel("funkcja kosztu")
plt.show()
