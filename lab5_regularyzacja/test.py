import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model as linm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

boston = datasets.load_boston()
X = boston.data
y = boston.target

print('X:\n', X)
print('y:\n', y)

#########################################
# Podział na zbiór treningowy i testowy (70-30%)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

#########################################
# Stworzenie obiektu

regr = linm.LinearRegression()

#########################################
# Uczenie modelu przy pomocy bazy treningowej

regr.fit(X_train, Y_train)

#########################################
# Przewidywanie wartości dla danych testowych

Y_pred = regr.predict(X_test)

#########################################
# Wyświetlenie parametrów prostej

print('Coefficients: \n', regr.coef_)

#########################################
#  Obliczamy rzeczywisty popełniony błąd średnio-kwadratowy
#
# print('Y_train',Y_train)
# print('Y_predicted',Y_pred)


# Effectiveness
print('Accuricy train: ', regr.score(X_train, Y_train))
print('Accuricy test: ', regr.score(X_test, Y_test))

scaler = StandardScaler()
scaler.fit(X)
scaler.mean_
X = scaler.transform(X)

# scaler = StandardScaler()
# scaler.fit(y)
# scaler.mean_
# X = scaler.transform(y)

steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', linm.LinearRegression())
]

pipe = Pipeline(steps)

pipe.fit(X_train, Y_train)

##########################################
## Uczenie modelu przy pomocy bazy treningowej

pipe.fit(X_train, Y_train)

#########################################
# Przewidywanie wartości dla danych testowych

Y_pred = pipe.predict(X_test)

#########################################
# Wyświetlenie parametrów prostej

# print('Coefficients: \n', regr.coef_)

#########################################
#  Obliczamy rzeczywisty popełniony błąd średnio-kwadratowy

# print('Y_train',Y_train)
# print('Y_predicted',Y_pred)


# Effectiveness
print('Accuricy train pipe: ', pipe.score(X_train, Y_train))
print('Accuricy test pipe: ', pipe.score(X_test, Y_test))

# reg_Ridge = Pipeline([('poly', PolynomialFeatures(degree=2)),
#                       ('model', linm.Ridge(alpha=10))])
#
# reg_Ridge.fit(X_train, Y_train)
# Y_pred = reg_Ridge.predict(X_test)
#
# print('Accuricy train reg_Ridge: ', reg_Ridge.score(X_train, Y_train))
# print('Accuricy test reg_Ridge: ', reg_Ridge.score(X_test, Y_test))
#
# reg_Lasso = Pipeline([('poly', PolynomialFeatures(degree=2)),
#                       ('model', linm.Lasso(alpha=0.1))])
#
# reg_Lasso.fit(X_train, Y_train)
# Y_pred = reg_Lasso.predict(X_test)
#
# print('Accuricy train reg_Lasso: ', reg_Lasso.score(X_train, Y_train))
# print('Accuricy test reg_Lasso: ', reg_Lasso.score(X_test, Y_test))
#
# deg_vect = []
# accuracy_train = []
# accuracy_test = []
# deg_max = 0
# deg_opt = 0
#
# for deg in range(3):
#     deg_vect = np.append(deg_vect, deg)
#     steps = [
#         ('poly', PolynomialFeatures(degree=deg)),
#         ('model', linm.LinearRegression())
#     ]
#
#     reg = Pipeline(steps)
#
#     reg.fit(X_train, Y_train)
#
#     accuracy_train = np.append(accuracy_train, reg.score(X_train, Y_train))
#     accuracy_test = np.append(accuracy_test, reg.score(X_test, Y_test))
#     if reg.score(X_test, Y_test) > deg_max:
#         deg_max = reg.score(X_test, Y_test)
#         deg_opt = deg
#
# plt.figure()
# plt.plot(deg_vect, accuracy_train)
# plt.title("Accuracy train linear")
# plt.xlabel('deg')
# plt.ylabel('accuracy')
#
# plt.figure()
# plt.plot(deg_vect, accuracy_test)
# plt.title("Accuracy test linear")
# plt.xlabel('alpha')
# plt.ylabel('accuracy')
# print("alpha optimal", deg_opt)
# print("score: ", deg_max)
#
# alph_vect = np.linspace(0, 100, 1000)
# accuracy_train = []
# accuracy_test = []
# score_max = 0
# alph_opt = 0
#
# for alph in alph_vect:
#     # reg_Ridge = Pipeline([('poly', PolynomialFeatures(degree=2)),
#     #                       ('model', linm.Ridge(alpha=alph))])
#
#     reg_Ridge = linm.Ridge(alpha=alph)
#     reg_Ridge.fit(X_train, Y_train)
#     accuracy_train = np.append(accuracy_train, reg_Ridge.score(X_train, Y_train))
#     accuracy_test = np.append(accuracy_test, reg_Ridge.score(X_test, Y_test))
#     # print('Accuricy train reg_Ridge: ', reg_Ridge.score(X_train, Y_train))
#     # print('Accuricy test reg_Ridge: ', reg_Ridge.score(X_test, Y_test))
#     if reg_Ridge.score(X_test, Y_test) > score_max:
#         score_max = reg_Lasso.score(X_test, Y_test)
#         alph_opt = alph
#
# plt.figure()
# plt.plot(alph_vect, accuracy_train)
# plt.title("Accuracy train Ridge")
#
# plt.figure()
# plt.plot(alph_vect, accuracy_test)
# plt.title("Accuracy test Ridge")
#
# print("alpha optimal", alph_opt)
# print("score: ", score_max)
#
# alph_vect = np.linspace(0, 1, 1000)
# accuracy_train = []
# accuracy_test = []
# score_max = 0
# alph_opt = 0
#
# for alph in alph_vect:
#     # reg_Lasso = Pipeline([('poly', PolynomialFeatures(degree=2)),
#     #                       ('model', linm.Lasso(alpha=alph))])
#
#     reg_Lasso = linm.Lasso(alpha=alph)
#
#     reg_Lasso.fit(X_train, Y_train)
#     accuracy_train = np.append(accuracy_train, reg_Lasso.score(X_train, Y_train))
#     accuracy_test = np.append(accuracy_test, reg_Lasso.score(X_test, Y_test))
#     if reg_Lasso.score(X_test, Y_test) > score_max:
#         score_max = reg_Lasso.score(X_test, Y_test)
#         alph_opt = alph
#
# print("alpha optimal", alph_opt)
# print("score: ", score_max)
#
# plt.figure()
# plt.plot(alph_vect, accuracy_train)
# plt.title("Accuracy train Lasso")
# plt.xlabel('alpha')
# plt.ylabel('accuracy')
#
# plt.figure()
# plt.plot(alph_vect, accuracy_test)
# plt.title("Accuracy test Lasso")
# plt.xlabel('alpha')
# plt.ylabel('accuracy')