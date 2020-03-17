# Artificial Neural Networks (ANN)

# Input Layer --> variables independientes
# Hidden Layers
# Output Layer --> variable dependiente

# Neurona

# Una neurona, tiene m entradas y una salida. A su vez, las salidas de unas neuronas se conectan con las entradas
# de otras. Cada una de las entradas de una neurona, tiene asociada un peso, que representa algo asi como la
# importancia que se le da a ese parametro, y estos pesos son los que se van ajustando a lo largo del proceso de
# aprendizaje. En funcion de las entradas y los pesos, la neurona calcula un valor, que es aplicado a una funcion de
# activacion (que define si la neurona se activa o no, o el peso de la misma). Al final, se da como resultado la salida.

# Funcion de Activacion

# Son las funciones a las que se le aplica la suma ponderara que calcula la neurona, para definir si se activa o no
# la neurona, o para calcular la salida. Algunas de las mas conocidas son: Treshold (escalon), Sigmoid (ideal para el
# calculo de probabilidades, Rectifier (rampa) y la Hiperbolic Tangent (similar a la sigmoid pero va de -1 a 1,
# cortando en el 0)

# Hidden Layer

# Cada neurona en las hidden layers, pueden tener en cuenta ciertas entradas y otras no, y en funcion de esas que
# tienen en cuenta y de los pesos asignados, calcular la suma ponderar y con la funcion de activacion definir si la
# misma se activa o no. Las salidas de cada neurona luego sirven como entradas o nuevos parametros para las de la
# siguiente capa.

# A la salida de la red, cuando se calcula la salida, esta es comparada con el valor objetivo, y en funcion de eso, se
# calcula el costo, es decir, el error. Esto se hace por cada nueva entrada (conjunto de variables independientes).
# Luego de que se hizo un Epoch, en base a los costos calculados por cada entrada, se calcula el costo total, y recien
# ahi, en base a ese costo se actualizan los pesos iniciales y se inicia un nuevo ciclo. Esto se conoce como
# Backpropagation. El objetivo es minimizar los costos. Gradient Descent se usa para el calculo de los pesos.

# Epoch --> Un Epoch, es una ronda o ciclo completo de la red a traves del training set completo (deben hacerse varios
#           para mejorar el aprendizaje de la red)

# Stochastic Gradient Descent

# El Gradient Descent, tiene el problema de que al entontrar un minimo, el mismo puede ser un minimo local, y nunca
# garantiza encontrar el minimo local. El Stochastic Gradient Descent, evita este problema, y la forma de hacerlo es que
# en vez de calcular los nuevos pesos luego de cada Epoch o ciclo, lo hace por cada nueva muestra. Ademas, es mas, el
# Stochastic Gradient Descent es mas rapido que el Batch Gradient Descent (batch porque se aplica al final para los
# valores provistos al final de cada Epoch o ciclo.

# Forwardpropagation --> ingresan las entradas, se calcula la salida y el costo
# Backpropagation --> el costo se propaga en sentido inverso y se ajustan los pesos simultaneamente, ya que por la forma
#                     en la que esta estructurado el algoritmo, es como que sabe que tanta culpa del error tiene cada
#                     un de los pesos en la red.


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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
from keras.layers import Dense # necesario para construir las capaz

# Inicializacion de la ANN
classifier = Sequential()

# Hidden Layers --> Rectifier activation function ('relu' en keras)
# Output Layer --> Sigmoid activation function (para obtener la probabilidad de abandono de cada cliente del banco)

# Una forma de definir la cantidad de nodos en la Hidden Layer, es tomar el promedio de la cantidad de nodos en la Input
# Layer y la Output Layer, en este caso hay 11 en la Input (por las 11 variables independientes) y 1 en la Output (ya
# que la variable de salida es binaria), por lo tanto, (11 + 1) / 2 = 6

# input_dim = cuantas entradas espera esa capa

classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu')) # first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu')) # second hidden layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # output layer

# Compilando la ANN

# optimizer = metodo de optimizacion de los pesos, en este caso, Adam es un Stochastic Gradent Descent optimizado
# loss = funcion que sirve de soporte al gradent descent para el calculo del error o costo
# metrics = criterio para medir la performance de la red
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenando la ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# accuracy del 86% aproximadamente