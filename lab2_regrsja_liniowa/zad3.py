import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, linear_model
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
# print description
# print(boston.DESCR)

boston_X = boston.data
boston_Y = boston.target

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_Y, test_size=0.3, random_state=1)

# Stworzenie obiektu
# regr = linear_model.LinearRegression()
# regr = linear_model.Ridge(alpha=.5)
# regr = linear_model.Lasso(alpha=5.1)
regr = linear_model.ElasticNet(alpha=.5, l1_ratio=0.5)

# Uczenie modelu przy pomocy bazy treningowej
regr.fit(X_train, y_train)
# Przewidywanie wartości dla danych testowych
y_pred = regr.predict(X_test)

# Wyświetlenie parametrów prostej
# print('Coefficients: \n', regr.coef_)
for idx, col_name in enumerate(boston.feature_names):
    print("The coefficient for {} is {}".format(col_name, regr.coef_[idx]))
print("b = ", regr.intercept_)

#  Obliczamy rzeczywisty popełniony błąd średnio-kwadratowy
error = np.mean((regr.predict(X_test) - y_test) ** 2)
print("Residual sum of squares: {}".format(error))

for i in range(12):
    plt.figure()
    plt.scatter(X_test[:, i], y_test, color='blue', linewidth=1)
    plt.xlabel(boston.feature_names[i])
    plt.ylabel('Price')
    plt.plot(X_test[:, i], regr.intercept_ + regr.coef_[i] * X_test[:, i], color='red', linewidth=1)
plt.show()
