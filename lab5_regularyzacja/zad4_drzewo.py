from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals.six import StringIO
import pydot

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)

depth = np.linspace(1, 10, 10, dtype=int)
accuracy_train = []
accuracy_test = []
depth_opt = 0
val_opt = 0

for depth_value in depth:
    clf = tree.DecisionTreeClassifier(max_depth=depth_value, random_state=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    accuracy_test.append(score)
    if score > val_opt:
        val_opt = score
        depth_opt = depth_value

plt.figure()
plt.plot(depth, accuracy_test)
plt.title("Dokładność zbioru testowego")
plt.xlabel('k')
plt.ylabel('Dokładność')
plt.show()

print("Optymalna głębokość drzewa = ", depth_opt)
print("Dokładność dopasowania opymalnego = ", val_opt)
