# SUPPORT VECTOR REGRESSION

# Traza una linea que modela la tendencia de los datos, llamada Hiperplano (como una linea de regression) y
# dos lineas a los costados, llamados Vectores de Soporte a una misma distancia del Hiperplano, donde la idea
# es que en toda esa area, se cubra la mayor cantidad de datos que queremos modelar. Para la ecuacion final
# son considerados los datos que quedan fuera, para los que se les calcula una la distancia que tienen de la
# banda mas cercana (Epsilon), que representa el error que puede cometerse.

# Se busca la linea que mejor modele la tendencia de los datos, la cual puede ser lineal o no lineal

# Para conjuntos de datos grandes, el tiempo de entrenamiento puede ser muy alto

# El kernel indica el tipo de funcion (linear, polynomial, rbf (gaussian), sigmoid)

# https://www.youtube.com/watch?v=wvLIcYk2kgQ (otra explicacion)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 2 - Regression/3. Polynomial Regression/Position_Salaries.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, 1:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

# feature scaling, o lo que es lo mismo que normalizacion (SVR no lo hace por defecto)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1, 1))[:, 0]

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# prediciendo un valor
# se normaliza el valor, se predice y se le hace la normalizacion inversa para obtener el valor en la escala original
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# grafica de la SVR
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Positions vs Salaries (SVR Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()