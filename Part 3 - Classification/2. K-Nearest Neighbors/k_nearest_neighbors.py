# K-NEAREST NEIGHBORS (K-NN)

# Clasifica el nuevo dato con la clase que predomina entre los N datos (vecinos) mas cercanos, segun la distancia
# Euclidea (pueden utilizarse otros tipos de distancias). Uno de los valores mas comunes para N es 5.

# Siendo dos puntos P1 = (X1; Y1) y P2 = (X2; Y2)
# La Distancia Euclidea entre ambos se calcula como --> sqrt((x2 - x1)^2 + (y2 - y1)^2))

# Es basicamente la hipotenusa del triangulo rectangulo formado por ambos puntos

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
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
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
# al ser un modelo de clasificacion NO lineal, el limite de prediccion no es una linea recta
from utils import plot_classification

plot_classification(X_train, y_train, classifier, 'K-NN (Training Set)', 'Age', 'Estimated Salary')

plot_classification(X_test, y_test, classifier, 'K-NN (Test Set)', 'Age', 'Estimated Salary')
