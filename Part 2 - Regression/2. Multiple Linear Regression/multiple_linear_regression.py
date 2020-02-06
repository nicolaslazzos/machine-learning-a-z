# MULTIPLE LINEAR REGRESSION

# Formula --> y = b0 + (b1 * x1) + (b2 * x2) + ... + (bn * xn)

# y =  variable dependiente (lo que queremos predecir)
# xn = variables independiente (las que hacen variar la variable dependiente)
# b0 = es una constante, indica en que punto la linea de regresion corta el eje y
# bn = indican en que proporcion un cambio en xn afecta a y
# n = cantidad de variables independientes

# es igual que la regresion simple, solo que como su nombre lo dice, incluye mas de una variable independiente

# Assumptions of a Linear Regression
#   1. Linearity
#   2. Homoscedasticity
#   3. Multivariate normality
#   4. Independence of errores
#   5. Lack of multicollinearity

# En el caso de las Dummy Variables, no es necesario incluir todas, ya que si por ejemplo tengo 2 tipos
# de estados (New York y California) incluyendo solo la columna de uno de los dos, el otro se obtiene por
# descarte, es decir, si no es New York, entonces es California. Si incluyeramos ambas variables, basicamente
# una estaria prediciendo a la otra, por lo que el modelo no podria entender bien la diferencia entre el efecto
# de una variable y la otra. Esto se llama Multillinearity. De todas las Dummy Variables que tengamos siempre
# debe omitirse una (por grupo de Dummy Variables).

# P-value --> es la probabilidad de error de aceptar la Hipotesis como cierta, y que en realidad sea falsa
#             es decir, que mientras menor sea, mejor. Debe ser menor al valor de significancia para aceptar
#             la hipotesis y rechazar la hipotesis nula. Por debajo del 0,025 por lo general.
#             En regresion, al analizar la correlacion entre una variable independiente y la variable
#             dependiente, indica la incertidumbre de esa correlacion, o la fuerza (mientras menor sea el valor)

# Que variables usar para entrenar el modelo?

# Backward Elimination --> establecer un nivel de significancia, entrenar el modelo con todas las variables y
#                          descartar aquellas que posean un P-value mayor al nivel establecido

# Forward Selection --> establecer un nivel de significancia, entrenar un modelo de regresion simple por cada variable
#                       independiente y elegir el que menor P-value tenga. Luego crear un modelo por cada combinacion
#                       de la variable elegida y las demas y seleccionar la que menor P-value tenga. Luego repetir
#                       el proceso hasta que se acaben las variables o ninguna este por debajo del P-value

# Bidirectional Elimination --> establecer dos niveles de significancia, uno para entrar al modelo y otro para quedarse.
#                               Se van agregando variables que tengan un P-value menor al nivel de significancia para
#                               entrar, segun el metodo del Forward Elimination. Y por cada variable que se agrega, se
#                               ejecuta un Backward Elimination teniendo en cuenta el nivel de significancia para
#                               quedarse, hasta que no se puede agregar ni quitar ninguna variable.

# Score Comparison --> construir todos los modelos posibles y elegir el mejor segun un cierto criterio.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 2 - Regression/2. Multiple Linear Regression/50_Startups.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, :-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

# encodear las variables categoricas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# remover una de las Dummy Variables (no es necesario hacerlo manual, la libreria ya lo hace)
X = X[:, 1:]

# separar el dataset en un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

# se entrena el modelo de regresion lineal con los datos de entrenamiento
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Backward Elimination (Manual)
import statsmodels.api as sm

# agregar la columna x0 al dataset
X = np.append(arr=np.ones((50, 1)), values=X, axis=1).astype('float64')

# Nivel de Significancia = 0.05

# matriz optima (de variables independientes significativas)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # Ordinary Least Squares
regressor_OLS.summary()

# se elimina x2 por mayor P-value
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# se elimina x1 por mayor P-value
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# se elimina x4 por mayor P-value
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# se elimina x5 por mayor P-value
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Backward Elimination (Automatic with P-value)
def backwardEliminationP(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardEliminationP(X_opt, SL)

# Backward Elimination (Automatic with P-value and Adjusted R-Squared)
def backwardEliminationPR(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardEliminationPR(X_opt, SL)