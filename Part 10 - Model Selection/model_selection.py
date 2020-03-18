# MODEL SELECTION

# Model Selection consiste en elegir los mejores hyperparametros (aquellos que se configuran, es decir, que no son
# aprendidos a partir de los datos) para nuestros modelos de machine learning.

# K-FOLD CROSS VALIDATION

# Lo que hace k-fold cross validation, es separar el training set en K subsets o "Folds" y utiliza K-1 folds para
# entrenar el modelo, y el restante para testearlo, y continua iterando de esta manera hasta que todos los folds fueron
# usados para test, es decir, se ejecutaron todas las combinaciones. De esta forma, se puede obtener la precision
# promedio a traves de todas las iteraciones asi como tambien la desviacion, para tener una idea de la varianza del
# modelo. Se obtiene una idea mucho mejor de la performance real del modelo a comparacion de entrenando una unica vez y
# calculando la precision en un unico test set.

# GRID SEARCH

# Grid Search nos va a ayudar a darnos cuenta si debemos aplicar un modelo lineal o no lineal, segun nuestro problema.
# Ademas nos va a ayudar a elegir los mejores hyperparametros.
# Debe ser aplicado luego de entrenar el modelo, ya que lo recibe como parametro. En este caso lo aplicamos luego de
# k-fold cross validation, asi una vez que evaluamos la preformance del modelo, lo mejoramos con grid search.

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
from sklearn.svm import SVC

# Gaussian kernel
classifier = SVC(kernel='rbf', random_state=0)
# classifier = SVC(kernel='rbf', C=1000, gamma=0.1, random_state=0) # with grid search
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# aplicando k-fold cross validation
from sklearn.model_selection import cross_val_score

# esta funcion calculara la accuracy para cada combinacion de folds de training y fold de test
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10) # cv = cantidad de folds
accuracies.mean() # precision promedio
accuracies.std() # desviacion estandar

# aplicando grid search
from sklearn.model_selection import GridSearchCV

# probando estas dos combinaciones de parametros, ya sabremos si debemos usar un modelo lineal o no lineal
# ademas de los valores optimos para los parametros especificados
parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}
]

# scoring = metrica en funcion de la cual se elegira el mejor modelo
# cv = cantidad de folds, ya que para el calculo de la metrica de scoring, grid search usa k-fold cross validation
# n_jobs = nucleos del procesador a usar
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# grafica de la clasificacion (regiones de prediccion)
# al estar usando el "rbf" kernel (Gaussian) que es no lineal, el limite de prediccion no es una linea recta
from utils import plot_classification

plot_classification(X_train, y_train, classifier, 'SVM Gaussian Kernel (Training Set)', 'Age', 'Estimated Salary')

plot_classification(X_test, y_test, classifier, 'SVM Gaussian Kernel (Test Set)', 'Age', 'Estimated Salary')