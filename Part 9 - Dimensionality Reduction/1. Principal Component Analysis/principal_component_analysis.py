# PRINCIPAL COMPONENT ANALYSIS (PCA)

# Es probablemente el algoritmo mas popular para reduccion de dimensiones. Es una tecnica de Feature Extraction.

# Busca encontrar correlaciones entre variables. Si hay una fuerte correlacion entre variables, esta puede proyectarse
# a una dimension menor y manteniendo la mayor cantidad de informacion. Lo que hace es, de las m variables
# independientes del dataset, toma p <= m nuevas variables independientes, que mejor explican la variacion o varianza
# del dataset, y esto, sin tener en cuenta la variable dependiente, lo que lo convierte en un modelo no supervisado.
# Esto tambien permite visualizar los datos cuando el dataset tiene mas de 2 variables independientes, ya que permite
# reducir el numero de dimensiones del dataset hasta un numero que sea posible de graficar.

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
from sklearn.decomposition import PCA

# seteando n_components como "None", genera todos los "principal components" para luego ver cuantos vamos a elegir
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# este vector contiene el porcentaje de varianza explicada por cada uno de los "principal components"
explained_variance = pca.explained_variance_ratio_
# en este caso, con los primeros dos principal components es suficiente para la clasificacion que queremos hacer
# por lo tanto, esos pasaran a ser nuestras variables independientes

# creando y entrenando el modelo
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) # aproximadamente 97% de precision en el test set

# grafica de la clasificacion (regiones de prediccion)
# al ser un modelo de clasificacion lineal, el limite de prediccion es una linea recta
from utils import plot_classification

plot_classification(X_train, y_train, classifier, 'Logistic Regression with PCA (Training Set)', 'PC1', 'PC2')

plot_classification(X_test, y_test, classifier, 'Logistic Regression with PCA (Test Set)', 'PC1', 'PC2')