import numpy as np 
import numpy.random as rnd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
np.random.seed()

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m ,1)


plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$", rotation = 0, fontsize = 18)

plt.axis([-3, 3, 0, 10])

# plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias= False)
x_poly = poly_features.fit_transform(x)
print(x_poly[0])

#Generamos el modelo de regresion lineal
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

print(lin_reg.intercept_, lin_reg.coef_)

x_new = np.linspace(-3, 3, 100).reshape(100, 1)
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)
plt.plot(x, y, "b.")

plt.plot(x_new, y_new, "r--", linewidth = 2, label = "Prediction")

plt.xlabel("$X_1$")
plt.ylabel("$Y$", rotation = 0, fontsize = 18)
plt.legend(loc = "upper left", fontsize = 14)
plt.axis([-3, 3, 0, 10])
plt.show()
