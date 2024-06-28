import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Leemos el excel con los datos
data = pd.read_csv('countries.csv')

# Creamos un subdataFrame que tenga solo los elementos de México
data_mex = data[data.country == 'Mexico']

# Asignamos los datos para posteriormente poder entrenar el modelo
# Extraemos la columna de los años
x = np.asanyarray(data_mex[['year']])
# Extraemos la columna de la esperanza de vida
y = np.asanyarray(data_mex[['lifeExp']])

# Creamos nuestro modelo matemático (en este caso una regresión lineal)
model = linear_model.LinearRegression()

# Entrenamos el modelo en base a las columnas extraídas del dataset
model.fit(x, y)

# Hacemos la predicción del modelo
ypred = model.predict(x)

# Generamos la gráfica con los puntos en base a la esperanza de vida y a los años
plt.scatter(x, y)
# Creamos una línea punteada roja para mostrar las predicciones del modelo
plt.plot(x, ypred, '--r')
plt.xlabel("Años")
plt.ylabel("Esperanza de vida")
plt.title("Regresion Lineal Simple / esperanza de vida")
# Mostramos la gráfica
plt.show()

print("Esperanza de vida en 2005", model.predict([[2005]]))
print("Esperanza de vida en 2019", model.predict([[2019]]))
print("Esperanza de vida en 3019", model.predict([[3019]]))
print("Esperanza de vida en -542", model.predict([[-542]]))

correlation = np.corrcoef(x.T, y.T)[0, 1]

print("Coeficiente de correlación:", correlation)
