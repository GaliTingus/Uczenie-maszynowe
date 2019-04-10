import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, linear_model
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

boston = datasets.load_boston()
boston_X = boston.data
boston_Y = boston.target
boston_X = StandardScaler().fit_transform(boston_X)
# boston_X = scaler.transform(boston_X)
# boston_X = (boston_X )/tmp.with_std
# boston_Y = (boston_Y )/tmp.with_std
# data_std = (boston - boston.mean()) / (boston.std())

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_Y, test_size=0.3, random_state=40)

# Stworzenie obiektu
regr = linear_model.LinearRegression()
# regr = linear_model.Ridge(alpha=.5)
# regr = linear_model.Lasso(alpha=5.1)
# regr = linear_model.ElasticNet(alpha=.5, l1_ratio=0.5)

# Uczenie modelu przy pomocy bazy treningowej
regr.fit(X_train, y_train)
# Przewidywanie wartości dla danych testowych
y_pred = regr.predict(X_test)

# Wyświetlenie błędó
print('Blad treningowy: {}'.format(regr.score(X_train, y_train)))
print('Blad testowy: {}'.format(regr.score(X_test, y_test)))

# scaler = StandardScaler()
# tmp = scaler.fit(X_train, y_train)
# print(scaler.fit(X_train, y_train))
steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
]

pipe = Pipeline(steps)

pipe.fit(X_train, y_train)

# Wyświetlenie błędó
print('Blad treningowy: {}'.format(pipe.score(X_train, y_train)))
print('Blad testowy: {}'.format(pipe.score(X_test, y_test)))
