# RANDOM FOREST REGRESSION

# Es un tipo de Ensemble Learning (cuando se juntan varios algoritmos para formar uno mas robusto).
# Lo que hace es, tomar un grupo aleatorio de muestras del training set y crear un arbol de decision en base
# a esos puntos y repetir ese procedimiento N veces, por loo que finalmente se obtiene un conjunto de N arboles
# de decision. Para predecir un valor, se predice el valor con cada uno de los N arboles de decision y finalmente
# se toma la media de esas predicciones.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 2 - Regression/3. Polynomial Regression/Position_Salaries.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, 1:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# grafica de la regresion
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Positions vs Salaries (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()