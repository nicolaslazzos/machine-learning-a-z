# DROPOUT REGULARIZATION

# Es una tecnica para reducir el overfitting. Podemos notar si tenemos overfitting, cuando la accuracy en el test set,
# es bastante menor que en el training set, es decir, el modelo se ajusto demasiado a los datos del training set.
# Tambien podemos notarlo cuando al implementar k-fold cross validation, nos da mucha desviacion.

# Lo que hace esta tecnica, es en cada iteracion de la ANN, deshabilita ciertas neuronas aleatoriamente para que no sean
# muy dependientes entre ellas, de esta forma se aprenden otras correlaciones independientes que evitan que las neuronas
# aprendan demasiado causando overfitting. Puede aplicarse a una capa o a varias. Mientras a mas capas de aplique, mas
# se reduce la probabilidad de overfitting.

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

# Implementacion de la ANN
import keras
from keras.models import Sequential # necesario para inicializar la ANN
from keras.layers import Dense, Dropout

# Inicializacion de la ANN
classifier = Sequential()

classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu')) # first hidden layer

# p = fraccion de neuronas a desactivar, se recomienda arrancar con 0.1 e ir probando si el overfitting continua, y en
# caso de ser asi, ir aumentando, pero no se recomienda ir mas alla de 0.5 ya que probablemente causara underfitting
classifier.add(Dropout(p=0.1))

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu')) # second hidden layer
classifier.add(Dropout(p=0.1))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # output layer

# Compilando la ANN
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenando la ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)