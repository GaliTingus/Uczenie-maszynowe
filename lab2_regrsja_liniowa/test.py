# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model as linm

# Reggression models
# http://scikit-learn.org/stable/modules/linear_model.html

# Load the diabetes dataset
boston = datasets.load_boston()
# print description
print(boston.DESCR)
# get the data
boston_X = boston.data
boston_Y = boston.target

# %% normalizacja zmiennych
X = (boston_X - boston_X.mean()) / (boston_X.std())
Y = (boston_Y - boston_Y.mean()) / (boston_Y.std())
# %% zad 1 
# podzielenie zbiorow
X_train = X[0:350, :]
Y_train = Y[0:350]
X_test = X[350:506, :]
Y_test = Y[350:506]
# Stworzenie obiektu 
regr = linm.LinearRegression()

# %% zad 2
# Uczenie modelu przy pomocy bazy treningowej
regr.fit(X_train, Y_train)
# Przewidywanie wartości dla danych testowych
Y_predicted = regr.predict(X_test)

# %% Wyświetlenie parametrów prostej
print('Coefficients: \n', regr.coef_)
#  Obliczamy rzeczywisty popełniony błąd średnio-kwadratowy
error = np.mean((regr.predict(X_test) - Y_test) ** 2)
print("Residual sum of squares: {}".format(error))

# %% zad 3
for i in range(X_test.shape[1]):
    plt.plot(X_test[:, i], X_test[:, i] * regr.coef_[i], 'r')
    plt.scatter(X_test[:, i], Y_test, c='b')
    plt.scatter(X_test[:, i], Y_predicted, c='y')
    plt.ylabel(boston.feature_names[i])
    plt.xlabel('Prize')
    plt.legend(['trend', 'tested', 'predicted'])
    plt.title('Prize(' + boston.feature_names[i] + ')')
    plt.show()

# %% model vs dane
print('porównanie danych z modelem')
plt.scatter(range(boston_X.shape[0]), Y)
plt.scatter(range(boston_X.shape[0]), regr.predict(X))
plt.show()

# %% porównania modeli regresji

reg_LinReg = linm.LinearRegression()
reg_Ridge = linm.Ridge(alpha=.5)
reg_Lasso = linm.Lasso(alpha=5.1)
reg_ElNet = linm.ElasticNet(alpha=.5, l1_ratio=0.5)

# Uczenie modelow przy pomocy bazy treningowej
reg_LinReg.fit(X_train, Y_train)
reg_Ridge.fit(X_train, Y_train)
reg_Lasso.fit(X_train, Y_train)
reg_ElNet.fit(X_train, Y_train)

# Przewidywanie wartości dla danych testowych
predict_Ridge = reg_Ridge.predict(X_test)
predict_LinReg = reg_LinReg.predict(X_test)
predict_Lasso = reg_Lasso.predict(X_test)
predict_ElNet = reg_ElNet.predict(X_test)

#  Obliczamy rzeczywisty popełniony błąd średnio-kwadratowy dla kazdego z modelow
error_Ridge = np.mean((reg_Ridge.predict(X_test) - Y_test) ** 2)
error_LinReg = np.mean((reg_LinReg.predict(X_test) - Y_test) ** 2)
error_Lasso = np.mean((reg_Lasso.predict(X_test) - Y_test) ** 2)
error_ElNet = np.mean((reg_ElNet.predict(X_test) - Y_test) ** 2)

print("Residual sum of squares for Ridge : {}".format(error_Ridge))
print("Residual sum of squares for LinReg : {}".format(error_LinReg))
print("Residual sum of squares for Lasso : {}".format(error_Lasso))
print("Residual sum of squares for ElNet : {}".format(error_ElNet))
