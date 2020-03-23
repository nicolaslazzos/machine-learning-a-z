# K-FOLD CROSS VALIDATION EN UNA ANN

# K-Fold Cross Validation es una tecnica que permite evaluar la precision del modelo de una mejor forma
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
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# KerasClassifier es un wrapper que nos va a permitir aplicar k-fold cross validation (que es de scikit-learn) sobre un
# modelo de Keras
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
accuracies.mean()  # una precision promedio del 84%
accuracies.std()  # no hay mucha desviacion entre las precisiones calculadas, lo cual es bueno