# LOGISTIC REGRESSION

# Formula --> Ln (p/(1-p)) = b0 + (b1 * x)

# En este caso, como hablamos de clasificacion, los valores a predecir, son categorias, valores discretos, no
# continuos como en la regresion lineal, por lo tanto, no puede usarse la linea de regresion para predecir las
# categorias. Por eso, a la linea de regresion, se le aplica una funcion Sigmoide, con lo que obtenemos una nueva
# funcion, que no va a predecir las categorias en si, sino la probabilidad de una muestra de pertenecer a una
# categoria o a otra.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 3 - Classification/1. Logistic Regression/Social_Network_Ads.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes (Gender and Estimated Salary)
X = dataset.iloc[:, 2:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

# separar el dataset un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling, o lo que es lo mismo que normalizacion
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# aca no hace falta el fit, porque ya se le hizo el fit a X antes
X_test = sc_X.transform(X_test)

# creando y entrenando el modelo
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# (0;0) predicciones de la clase 0 correctas
# (1;0) predicciones de la clase 0 incorrectas (eran de la clase 1)
# (0;1) predicciones de la clase 1 incorrectas (eran de la clase 0)
# (1;1) predicciones de la clase 1 correctas

# grafica de la clasificacion (regiones de prediccion)
# al ser un modelo de clasificacion lineal, el limite de prediccion es una linea recta
from utils import plot_classification

plot_classification(X_train, y_train, classifier, 'Logistic Regression (Training Set)', 'Age', 'Estimated Salary')

plot_classification(X_test, y_test, classifier, 'Logistic Regression (Test Set)', 'Age', 'Estimated Salary')