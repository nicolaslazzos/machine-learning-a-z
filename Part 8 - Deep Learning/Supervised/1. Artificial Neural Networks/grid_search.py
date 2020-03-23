# GRID SEARCH EN UNA ANN

# Consiste en encontrar los valores optimos de los hiperparametros (aquellos que no son aprendidos), es decir,
# encontrar los mejores valores, los que produzcan la mayor precision en el modelo.
# (Ver seccion "Model Selection & Boosting")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Caso de estudio --> Churn en un banco
dataset = pd.read_csv('Part 8 - Deep Learning/Supervised/1. Artificial Neural Networks/Churn_Modelling.csv')

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

# feature scaling, o lo que es lo mismo que normalizacion
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential  # necesario para inicializar la ANN
from keras.layers import Dense  # necesario para construir las capa
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# KerasClassifier es un wrapper que nos va a permitir aplicar el grid search (que es de scikit-learn) sobre un
# modelo de Keras
classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
