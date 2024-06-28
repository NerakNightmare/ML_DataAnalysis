import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets

iris = datasets.load_iris()
x = iris["data"][:, (2,3)]

y = iris["target"]

plt.plot(x[y == 2, 0], x[y==2,1], "g^",label = "iris-virginica")
plt.plot(x[y == 1, 0], x[y==1,1], "bs",label = "iris-versicolor")
plt.plot(x[y == 0, 0], x[y==0,1], "yo",label = "iris-setosa")

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

softmax = LogisticRegression(multi_class="multinomial",
                             solver="lbfgs", C=10)

softmax.fit(xtrain, ytrain)

print("Train: ", softmax.score(xtrain, ytrain))
print("Test: ", softmax.score(xtest, ytest))
plt.show()