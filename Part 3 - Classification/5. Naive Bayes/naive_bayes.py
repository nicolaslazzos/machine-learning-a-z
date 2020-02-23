# NAIVE BAYES

# Teorema de Bayes

# Formula --> P(A|B) = (P(B|A) * P(A)) / P(B)
# P(A|B) = Probabilidad de A dado que B

# Lo que hace el algoritmo Naive Bayes, es aplicar el teorema de bayes para cada posible categoria en funcion de las
# features, es decir, calcula las probabilidades de que una muestra sea de cada categoria, en funcion de los valores
# de las variables independientes para finalmente compararlas y asignar la muestra a una categoria.
# Por ejemplo si tenemos Categoria1 y Categoria2, para una muestra se calcula la P(Cat1|X) y P(Cat2|X) siendo X el
# valor correspondiente a la muestra en funcion de sus variables independientes.

# Para el calculo de cada termino ver video "Naive Bayes Intuition"

# Calculo de los terminos:
#       P(Cat1|X) --> Posterior Probability: se calcula aplicando la formula.
#       P(X|Cat1) --> Likelihood: se elige un radio alrededor de X y se calcula como la cantidad de muestras dentro
#                                 de ese radio de Cat1 sobre la cantidad Total de muestras de Cat1.
#       P(X) --> Marginal Likelihood: se elige un radio alrededor de X y se calcula como la cantidad de muestras dentro
#                                     de ese radio, sobre el Total de muestras. Este valor es el siempre el mismo para
#                                     la muestra en cuestion, por lo que la comparacion entre las categorias puede
#                                     hacerse directamente prescindiendo de ese valor y se ahorra un calculo.
#       P(Cat1) --> Prior Probability: cantidad de muestras de la Cat1 sobre el Total de muestras.

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
from sklearn.naive_bayes import GaussianNB

# Gaussian kernel
classifier = GaussianNB()
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

plot_classification(X_train, y_train, classifier, 'Naive Bayes (Training Set)', 'Age', 'Estimated Salary')

plot_classification(X_test, y_test, classifier, 'Naive Bayes (Test Set)', 'Age', 'Estimated Salary')