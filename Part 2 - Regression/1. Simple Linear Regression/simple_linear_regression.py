# SIMPLE LINEAR REGRESSION

# Formula --> y = b0 + (b1 * x1)

# y =  variable dependiente (lo que queremos predecir)
# x1 = variable independiente (la que hace variar la variable dependiente)
# b0 = es una constante, indica en que punto la linea de regresion corta el eje y
# b1 = indica en que proporcion un cambio en x afecta a y, es la pendiente de la linea de regresion

# Ordinary Least Squares

# para encontrar la mejora linea de regresion, se calcula la suma de los errores al cuadrado
# o el error cuadratico medio (MSE) y se elige la recta que minimiza estos errores

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Part 2 - Regression/1. Simple Linear Regression/Salary_Data.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, :-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

# separar el dataset en un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
# el modelo va a aprender la correlacion entre X_train y y_train para despues poder predecir y en base a X_test

# feature scaling (normalizacion) no hace falta con la regresion lineal debido a que la libreria ya se encarga de eso

from sklearn.linear_model import LinearRegression

# se entrena el modelo de regresion lineal con los datos de entrenamiento
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# grafica training dataset
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# grafica test dataset
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()