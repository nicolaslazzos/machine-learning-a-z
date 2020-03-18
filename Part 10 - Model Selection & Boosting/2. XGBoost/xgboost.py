# XGBOOST

# Es la implementacion mas poderosa de Grdient Boosting en terminos de performance de los modelos y velocidad de
# ejecucion. Gradient Boosting es una tecnica de machine learning utilizada en problemas de regresion y clasificacion
# que produce un modelo de prediccion en la forma de un conjunto de modelos de prediccion debiles, generalmente arboles
# de decision. Ademas no requiere normalizacion, por lo que podemos conservar la interpretacion de los datos.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Caso de estudio --> Churn en un banco
dataset = pd.read_csv('Part 8 - Deep Learning/1. Artificial Neural Networks/Churn_Modelling.csv')

# matriz de variables independientes
X = dataset.iloc[:, 3:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

# encodear las variables categoricas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# se transforma cada categoria a un numero
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1, 2])], remainder='passthrough')
X = ct.fit_transform(X)

# removiendo una dummy variable de cada variable categorica
X = np.delete(X, [0, 3], 1)

# separar el dataset un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# implementando XGBoost
from xgboost import XGBClassifier

classifier = XGBClassifier()
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