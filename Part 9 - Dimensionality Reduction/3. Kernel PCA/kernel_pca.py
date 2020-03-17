# KERNEL PCA

# Tanto PCA y LDA, vistos anteriormente, funcionan en problemas lineales, es decir, cuando los datos son linealmente
# separables. En este caso, Kernel PCA es una tecnica de reduccion de dimensiones que puede utilizarse para problemas
# no lineales.

# En el problema que se resuelve a continuacion, al ser no lineal, si aplicamos Logistic Regression, no se obtiene una
# muy buena clasificacion, ya que las regiones de clasificacion estan separadas por una linea recta. Con Kernel PCA, lo
# que nos permite es reducir la dimension del dataset, extrayendo los "Principal Components", y no solo eso, sino que
# estos nuevos componentes seran linealmente separables (al mapear los datos a un espacio de dimensiones superior donde
# los mismos son linealmente separables).
# (Ver video "Mapping to a higher dimension")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# usuarios que compraron o no un auto en funcion de una publicidad en redes sociales
dataset = pd.read_csv('Part 3 - Classification/1. Logistic Regression/Social_Network_Ads.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes
X = dataset.iloc[:, 2:-1].values

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

# Dimensionality Reduction (Aplicando Kernel PCA)
from sklearn.decomposition import KernelPCA

# seteando n_components como "None", genera todos los "principal components" para luego ver cuantos vamos a elegir
# en este caso vamos a usar el kernel Gaussian = rbf
kernel_pca = KernelPCA(n_components=2, kernel='rbf')
X_train = kernel_pca.fit_transform(X_train)
X_test = kernel_pca.transform(X_test)

# creando y entrenando el modelo
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) # 90% de precision en el test set

# grafica de la clasificacion (regiones de prediccion)
# al ser un modelo de clasificacion lineal, el limite de prediccion es una linea recta
from utils import plot_classification

plot_classification(X_train, y_train, classifier, 'Logistic Regression with PCA (Training Set)', 'PC1', 'PC2')

plot_classification(X_test, y_test, classifier, 'Logistic Regression with PCA (Test Set)', 'PC1', 'PC2')

