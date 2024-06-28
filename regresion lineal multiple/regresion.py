import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns 

data = pd.read_csv("diabetes.csv")

pd.plotting.scatter_matrix(data)

corr = data.corr()

plt.figure()
#sacar la correlacion entre los datos
sns.heatmap(corr, xticklabels=corr.columns,
           yticklabels=corr.columns)

#elegir variables

x = np.asanyarray(data.drop(columns = ["Outcome"]))
y = np.asanyarray(data[["Outcome"]])


#Train/Test Split

xtrain, xtest, ytrain, ytest = train_test_split(x,y)


#Crear modelo
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression())])

#Entrenar modelo
model.fit(xtrain, ytrain.ravel())

#calculamos le desempeño
#en este modelo de clasificacion, nos muestra a cuantos le atino
print("Train: ", model.score(xtrain, ytrain.ravel()))
print("Test: ", model.score(xtest, ytest.ravel()))

plt.figure()
#explicar variables, que tanto me afectan las variables
coeff = np.abs(model.named_steps["logit"].coef_[0])
coeff = coeff / np.sum(coeff)
labels = list(data.drop(columns = ["Outcome"]).columns)
features = pd.DataFrame()
features["coeff"] = coeff
features["features"] = labels
features.sort_values(by="coeff", ascending=True, inplace=True)
features.set_index("features", inplace=True)
features.coeff.plot(kind="barh")
plt.xlabel('Importance')
plt.show()

