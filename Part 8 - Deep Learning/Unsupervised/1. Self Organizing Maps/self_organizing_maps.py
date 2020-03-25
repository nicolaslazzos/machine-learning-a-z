# SELF ORGANIZING MAPS (SOM)

# Son usados para Feature Detection, para reducir dimensiones y por lo tanto permite representar los resultados
# graficamente. Permite realizar clustering con muchas dimensiones de entrada y visualizar el resultado graficamente.

# En los SOM, se tiene tantas entradas como variables independientes, y se tienen muchas neuronas de salida, todas
# interconectadas con las de entrada. No hay funciones de activacion, y ademas, los pesos de cada neurona representan
# algo asi como puntos en el espacio de coordenadas de los datos que estamos analizando, es decir, se tiene tantos
# pesos como dimensiones tiene el dataset. Y en este tipo de redes, los pesos no se multiplican por las entradas, sino
# que las neuronas de salida compiten entre ellas, para ver cual de todas posee el punto que mas se acerca al punto que
# la red esta recibiendo como entrada. Ese neurona o punto mas cercano se denomina Best Matching Unit (BMU). Luego de
# cada itereacion, los pesos del BMU se actualizan para que coincidan mas con esa entrada. De la misma forma, los pesos
# de las neuroas que esten en un cierto radio (las neuronas de salida de un SOM suelen representarse como una matriz de
# n x m neuronas) al BMU, se actualizaran para acercarse a esa entrada, aunque en menor proporcion mientras mas nos
# alejamos el BMU. Los BMU tambien son conocidos como Winning Nodes.

# Por cada epoch, el radio que abarca cada BMU, va reduciendose, de forma que al actualizarse los pesos, van teniendo
# impacto sobre una menor cantidad de puntos o neuronas a su alrededor. Asi el proceso va volviendose cada vez mas
# preciso, realizando ajustes mas pequeños o especificos. Al ser un tipo de red de aprendizaje no supervisado, no
# efectua backpropagation, ya que no tiene un objetivo con el cual comprar.

# Caso de Estudio

# Vamos a aplicar RNN para un caso de deteccion de fraudes en aplicaciones para tarjetas de credito. Los casos de fraude
# seran aquellos que tengan anomalias por asi decirlo, y corresponden a las neuronas que se encuentran mas alejadas de
# sus vecinos, ya que han tenido pocas coincidencias. Entonces luego, para identificar cuales clientes son
# potencialmente fraudulentos, se buscara el cliente asociado a cada una de estas neuronas. Estas neuronas seran las que
# mayor Mean Interneuron Distance (MID), la cual es la media de las distancias de esta neurona a sus vecinas, dentro del
# radio definido inicialmente (sigma).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando el dataset
dataset = pd.read_csv('Part 8 - Deep Learning/Unsupervised/1. Self Organizing Maps/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Implementando el SOM
from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5, random_seed=4)
som.random_weights_init(X) # inicializando los pesos con valores aleatorios cercanos a 0
som.train_random(data=X, num_iteration=100)

# Visualizando los resultados
from pylab import bone, pcolor, colorbar, plot, show

bone() # inicializa la ventana del grafico

# distance_map() retona una matriz con las MID de cada neurona
pcolor(som.distance_map().T)
colorbar()

# Vamos a recorrer todos los clientes para identificar su nodo o neurona correspondiente, marcandola segun si realmente
# fue fraudulento o no, para ver la precision o relacion con el resultado del SOM
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor='None', markersize=10, markeredgewidth=2)

show()

# Identificando clientes que fueron aceptados pero que en base al SOM son potenciales fraudes
mappings = som.win_map(X) # obtiene el mapeo de clientes a nodos
frauds = np.concatenate((mappings[(2, 3)], mappings[(6, 1)]), axis=0) # son las coordenadas de los nodos alejados
frauds = scaler.inverse_transform(frauds) # potenciales clientes fraudulentos

# HYBRID DEEP LEARNING MODEL

# Tomaremos los resultados del SOM y los aplicaremos como entradas a una ANN, que nos dara las probabilidades de que los
# clientes obtenidos como potenciales fraudulentos, lo sean realmente o no.

# Matriz de variables independientes, incluyendo los resultados reales de si los clientes cometieron fraude o no
customers = dataset.iloc[:, 1:].values

# Creamos la variable independiente, poniendo en 1 los clientes que son potencialmente fraudulentos segun el SOM
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)

# Implementacion de la ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializacion de la ANN
classifier = Sequential()

classifier.add(Dense(units=2, input_dim=customers.shape[1], kernel_initializer='uniform', activation='relu')) # hidden layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # output layer

# Compilando la ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenando la ANN
classifier.fit(customers, is_fraud, batch_size=1, epochs=2) # al haber pocos datos, con 2 epochs basta

# Prediciendo las probabilidades de de fraude
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]