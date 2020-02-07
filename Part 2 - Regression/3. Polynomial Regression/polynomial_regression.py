# POLYNOMIAL REGRESSION

# Formula --> y = b0 + (b1 * x1) + (b2 * x1^2) + ... + (bn * x1^n)

# y =  variable dependiente (lo que queremos predecir)
# x1 = variable independiente (la que hace variar la variable dependiente)
# b0 = es una constante, indica en que punto la linea de regresion corta el eje y
# bn = indica en que proporcion un cambio en la potencia n de x afecta a y
# n = es el grado del polinomio

# se usa cuando hay una relacion no lineal entre la variable independiente y la dependiente

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 2 - Regression/3. Polynomial Regression/Position_Salaries.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, 1:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression

# modelo de regresion lineal para comparacion
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# modelo de regresion polinomica
from sklearn.preprocessing import PolynomialFeatures

# PolynomialFeatures lo que hace es crear cada una de las potencias de x segun el grado indicado,
# es decir, transforma la matriz lineal de features en una matriz polinomica
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# grafica de la regresion lineal
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Positions vs Salaries (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# grafica de la regresion polnomica
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Positions vs Salaries (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()