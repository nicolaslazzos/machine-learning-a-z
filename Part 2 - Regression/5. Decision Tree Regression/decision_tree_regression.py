# DECISION TREE REGRESSION

# Se crea un arbol de decision que separa los datos en un conjunto de grupos con valores relativamente similares.
# Al introducir un nuevo valor para predecir, dicha prediccion corresponde a la media de los valores de las
# muestras o datos de entrenamiento pertenecientes al mismo grupo (ver video para demostracion grafica).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 2 - Regression/3. Polynomial Regression/Position_Salaries.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, 1:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# grafica de la regresion
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Positions vs Salaries (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()