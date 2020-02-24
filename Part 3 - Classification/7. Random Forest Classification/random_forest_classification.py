# RANDOM FOREST CLASSIFICATION

# Es un tipo de Ensemble Learning (cuando se juntan varios algoritmos para formar uno mas robusto).
# Lo que hace es, tomar un grupo aleatorio de muestras del training set y crear un arbol de decision en base
# a esos puntos y repetir ese procedimiento N veces, por lo que finalmente se obtiene un conjunto de N arboles
# de decision. Para predecir un valor, se predice el valor con cada uno de los N arboles de decision y finalmente
# se asigna la nueva muestra a la clase que predomina entre las predicciones de los arboles, es decir, la que gana
# por mayoria.

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

# Feature scaling en este caso no se requiere, debido a que no es un algoritmo que se basa en la distancia euclidea,
# pero en este caso lo hacemos por el tipo de grafico que estamos usando. Si fuera a representar el arbol de decisiones,
# convendria no aplicar feature scaling, ya que no es necesario y asi poder observar los valores reales
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# aca no hace falta el fit, porque ya se le hizo el fit a X antes
X_test = sc_X.transform(X_test)

# creando y entrenando el modelo
from sklearn.ensemble import RandomForestClassifier

# Maxima profundidad de los arboles limitada a 2 para evitar overfitting
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, max_depth=2)
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
from utils import plot_classification

plot_classification(X_train, y_train, classifier, 'Random Fores Classification (Training Set)', 'Age', 'Estimated Salary')

plot_classification(X_test, y_test, classifier, 'Random Forest Classification (Test Set)', 'Age', 'Estimated Salary')