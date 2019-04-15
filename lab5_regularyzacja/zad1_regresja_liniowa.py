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
X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_Y, test_size=0.3)

# Stworzenie obiektu
regr = linear_model.LinearRegression()
# regr = linear_model.Ridge(alpha=.5)
# regr = linear_model.Lasso(alpha=5.1)
# regr = linear_model.ElasticNet(alpha=.5, l1_ratio=0.5)

# Uczenie modelu przy pomocy bazy treningowej
regr.fit(X_train, y_train)
# Przewidywanie wartości dla danych testowych
y_pred = regr.predict(X_test)

# Wyświetlenie błędów
print('Blad treningowy reg_lin: {}'.format(regr.score(X_train, y_train)))
print('Blad testowy reg_lin: {}'.format(regr.score(X_test, y_test)))

scaler = StandardScaler()
scaler.fit(boston_X)
X = scaler.transform(boston_X)

steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', linear_model.LinearRegression())
]

pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
pipe.fit(X_train, y_train)

Y_pred = pipe.predict(X_test)

# Wyświetlenie błędów
print('\nBlad treningowy reg_pipe_deg2: ', pipe.score(X_train, y_train))
print('Blad testowy reg_pipe_deg2: ', pipe.score(X_test, y_test))

######### RIDGE ############
reg_Ridge = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('model', linear_model.Ridge(alpha=10))])

reg_Ridge.fit(X_train, y_train)
Y_pred = reg_Ridge.predict(X_test)

print('\nBlad treningowy reg_Ridge: ', reg_Ridge.score(X_train, y_train))
print('Blad testowy reg_Ridge: ', reg_Ridge.score(X_test, y_test))

alph_vect = np.linspace(0, 200, 2000)
dokladnosc_train = []
dokladnosc_test = []

for alph in alph_vect:
    reg_Ridge = linear_model.Ridge(alpha=alph)
    reg_Ridge.fit(X_train, y_train)
    dokladnosc_train = np.append(dokladnosc_train, reg_Ridge.score(X_train, y_train))
    dokladnosc_test = np.append(dokladnosc_test, reg_Ridge.score(X_test, y_test))

plt.figure()
plt.plot(alph_vect, dokladnosc_train)
plt.title("Blad treningowy Ridge")
plt.xlabel('alpha')
plt.ylabel('dokładność')

plt.figure()
plt.plot(alph_vect, dokladnosc_test)
plt.title("Blad testowy Ridge")
plt.xlabel('alpha')
plt.ylabel('dokładność')

########## LESSO ##############
reg_Lasso = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('model', linear_model.Lasso(alpha=0.1))])

reg_Lasso.fit(X_train, y_train)
Y_pred = reg_Lasso.predict(X_test)

print('\nBlad treningowy reg_Lasso: ', reg_Lasso.score(X_train, y_train))
print('Blad testowy reg_Lasso: ', reg_Lasso.score(X_test, y_test))

alph_vect = np.linspace(0.001, 1, 1000)
dokladnosc_train = []
dokladnosc_test = []
score_max = 0
alph_opt = 0

for alph in alph_vect:
    reg_Lasso = linear_model.Lasso(alpha=alph)

    reg_Lasso.fit(X_train, y_train)
    dokladnosc_train = np.append(dokladnosc_train, reg_Lasso.score(X_train, y_train))
    dokladnosc_test = np.append(dokladnosc_test, reg_Lasso.score(X_test, y_test))
    if reg_Lasso.score(X_test, y_test) > score_max:
        score_max = reg_Lasso.score(X_test, y_test)
        alph_opt = alph

print("optymana alpha", alph_opt)
print("wynik: ", score_max)

plt.figure()
plt.plot(alph_vect, dokladnosc_train)
plt.title("Blad treningowy Lasso")
plt.xlabel('alpha')
plt.ylabel('dokładność')

plt.figure()
plt.plot(alph_vect, dokladnosc_test)
plt.title("Blad testowy Lasso")
plt.xlabel('alpha')
plt.ylabel('dokładność')
plt.show()
