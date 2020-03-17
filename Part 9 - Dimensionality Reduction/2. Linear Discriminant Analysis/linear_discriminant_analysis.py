# LINEAR DISCRIMINANT ANALYSIS (LDA)

# Es una tecnica de Feature Extraction. Bastante similar a PCA.

# Es usado en problemas de clasificacion, y busca reducir el numero de dimensiones pero conservando toda aquella
# informacion que nos sirve para discriminar entre clases. Lo que hace es, de las n variables independientes del
# dataset, toma k <= n-1 nuevas variables independientes o ejes, que maximizan la separacion entre las diferentes
# clases. A diferencia de PCA, LDA si utiliza la variable dependiente para analizar la relacion de esta ultima con cada
# una de las variables independientes, por lo que es un algoritmo supervisado.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# segmentos de clientes segun el tipo de vino, para saber que vino recomendar a cada cliente
dataset = pd.read_csv('Part 9 - Dimensionality Reduction/1. Principal Component Analysis/Wine.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, 0:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

# separar el dataset un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling, o lo que es lo mismo que normalizacion
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Dimensionality Reduction (Aplicando PCA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# seteando n_components como "None", genera todos los "principal components" para luego ver cuantos vamos a elegir
lda = LDA(n_components=None)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# creando y entrenando el modelo
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) # 100% de precision en el test set

# grafica de la clasificacion (regiones de prediccion)
# al ser un modelo de clasificacion lineal, el limite de prediccion es una linea recta
from utils import plot_classification

plot_classification(X_train, y_train, classifier, 'Logistic Regression with PCA (Training Set)', 'LD1', 'LD2')

plot_classification(X_test, y_test, classifier, 'Logistic Regression with PCA (Test Set)', 'LD1', 'LD2')