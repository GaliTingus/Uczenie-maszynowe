import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Wczytanie danych
path = 'breast_cancer.txt'
dataset = pd.read_csv(path, header=None,
                      names=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                             'Normal Nucleoli', 'Mitoses', 'Class'])

dataset['Class'].replace(2, 0, inplace=True)
dataset['Class'].replace(4, 1, inplace=True)

# Uzupełnienie brakujących danych mediana
median = dataset.median()
for i in dataset:
    dataset[i].fillna(median[i], inplace=True)

# Podział na X i y
X = dataset.drop(['ID', 'Class'], axis=1)
y = dataset['Class']

# Podział na zbiory uczące i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Obiekt regresji logistycznej L1
reg_Log = LogisticRegression(penalty='l1', C=1, solver='saga', max_iter=1000)
reg_Log.fit(X_train, y_train)

print('Dokładność zbioru treningowego reg_Log: ', reg_Log.score(X_train, y_train))
print('Dokładność zbioru testowego reg_Log: ', reg_Log.score(X_test, y_test))

c = np.linspace(0.0001, 1, 10)
accuracy_train = []
accuracy_test = []

# Pętla po wartościach C
for c_value in c:
    LogisticRegression(penalty='l1', C=c_value, solver='saga', max_iter=10000)
    reg_Log.fit(X_train, y_train)

    accuracy_train = np.append(accuracy_train, reg_Log.score(X_train, y_train))
    accuracy_test = np.append(accuracy_test, reg_Log.score(X_test, y_test))

plt.figure()
plt.plot(c, accuracy_train)
plt.title("Dokładność zbioru treningowego reg_Log")
plt.xlabel('C')
plt.ylabel('Dokładność')

plt.figure()
plt.plot(c, accuracy_test)
plt.title("Dokładność zbioru testowego reg_Log")
plt.xlabel('C')
plt.ylabel('Dokładność')
plt.show()
