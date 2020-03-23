# RECURRENT NEURAL NETWORKS

# Son un tipo de redes neuronales muy utilizadas de aprendizaje supervisado.
# Para hacer una analogia con el funcionamiento del cerebro y ver donde estamos parados, los pesos en una ANN, una vez
# entrenada, permanecen iguales, y procesaran la entrada de la misma forma mientras permanezcan asi, por lo que pueden
# tomarse como recuerdos, lo que corresponde al Lobulo Temporal del cerebro.
# Luego una CNN, se relaciona con la vision, las imagenes, el reconocimiento, por lo que corresponderia al Lobulo
# Occipital. Por ultimo, lo que vamos a ver ahora, es decir, las RNN, corresponderian al Lobulo Frontal, ya que se
# encarga de recuerdos a corto plazo, cosas que han pasado hace poco. Por ejemplo al traducir una oracion, se se hace
# palabra por palabra, se necesita saber de las palabras anteriores para traducir cada una de ellas y que tenga sentido.
# Para generar comentarios de lo que pasa en una pelicula por ejemplo, se necesitaria saber que va ocurriendo, conocer
# el contexto de la pelicula. Otro ejemplo es la prediccion de texto en los teclados de los smartphones, en base a lo
# que vamos escribiendo y los patrones que ha aprendido.

# Para hacer eso, en las RNN, las neuronas de las hidden layers, poseen un loop temporal, es decir, tienen una conexion
# a si mismas que les permite darse un auto-feedback y de esta forma, saber que ha ocurrido en iteraciones anteriores.
# Se van como guardando los distintos estados en el tiempo de una neurona.

# The Vanishing Gradient Problem

# Este problema se refiere a que, en la backpropagation, cuando se calcula la funcion de costo y se actualiza el peso de
# las neuronas, en una RNN, al tener un loop las neuronas, tambien deben actualizarse los pesos de los estados
# anteriores de las neuronas (dependiendo de a cuantos pasos atras estemos mirando) y al ser valores cercanos a cero,
# mientras mas atras vamos en el tiempo, al ir multiplicando la salida de la neurona, por el peso, vamos obteniendo
# valores cada vez mas chicos y se vuelve mas dificil o mas lento actualizar los pesos y que la red llegue a converger.
# Si los valores son muy grandes, sucede lo contrario, se van haciendo rapidamente cada vez mas grandes y esto se
# conoce como Exploding Gradient. Hay varias soluciones para estos problemas, pero la mejor es la llamada
# Long Short-Term Memory Networks (LSTMs)

# Long Short-Term Memory (LSTM)

# El problema era, que si el Wrec (peso recurrente, el que vuelve hacia atras para ajustar los pesos de los estados
# anteriores de las neuronas) era menor a 1, se volvia cada vez mas chico (Vanishing Gradient) y si era mayor a 1, se
# iba haciendo cada vez mas grande (Exploding Gradient), por lo tanto, lo que hace el metodo LSTM, es establecer que
# el Wrec, sea 1. (Buscar mejor explicacion)

# Una neurona de una red de este tipo, tiene 3 entradas (la propia entrada, la salida de la neurona anterior, y la
# memoria de la neurona anterior) y tiene 2 salidas (la propia salida y su memoria). Todas las entradas tienen la forma
# de un vector de valores. Cada neurona, al recibir las memorias, puede conservarlas o actualizarlas (reemplazando la
# memoria que se tiene, por una nueva) dependiendo de las entradas. Hay distintas variaciones disponibles de
# Long Short-Term Memories.
# (Ver video "LSTMs" y "LSTMs Variations" para entender mejor el funcionamiento)

# Caso de Estudio

# Prediciendo la tendencia del precio de las acciones de google. El modelo se va a entrenar con los datos de 5 años de
# datos de sobre el precio de las acciones de google durante cada dia hasta 2016 inclusive, para luego predecir las
# tendencias en el primer mes del año 2017.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando el dataset
dataset_train = pd.read_csv('Part 8 - Deep Learning/Supervised/3. Recurrent Neural Networks/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Para una RNN se recomienda la normalizacion de los datos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)

# Vamos a crear una estructura de datos con 60 timesteps y 1 salida, es decir, por cada nuevo precio, va a recordar los
# 60 anteriores. Esta estructura cuenta de 2 entidades, X_train, que tiene conjuntos de 60 precios y y_train, que por
# cada conjunto de X_train, posee el precio de las acciones el dia siguiente a los 60 dias.
X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Se pueden añadir mas dimensiones a los datos de X_train, que funcionen como indicadores para ayudar a la prediccion,
# pero en este caso unicamente usaremos el precio de apertura de las acciones y por eso, ingresamos 1 en el tercer
# parametero del reshape, ya que tenemos solo 1 dimension. (cantidad muestras, timesteps, dimension)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Implementando la RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Inicializacion de la RNN
regressor = Sequential() # regressor porque estamos prediciendo valores continuos

# El parametro units, es el numero de celdas LSTM o unidades de memoria que tendra la capa.
# Se setea return_sequences igual a True ya que va a tener capas LSTM apiladas. La ultima LSTM lo tendra seteado False.
# Por ultimo input_shape se refiere al formato de las entradas, indicando los timesteps y la cantidad de indicadores.

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))) # primer LSTM layer
regressor.add(Dropout(rate=0.2)) # desactiva 20% de neuronas aleatoriamente en cada iteracion para evitar overfitting

# No hace especificar el input_shape ya porque lo reconoce automaticamente de la capa anterior
regressor.add(LSTM(units=50, return_sequences=True)) # segunda LSTM layer
regressor.add(Dropout(rate=0.2)) # segundo dropout

regressor.add(LSTM(units=50, return_sequences=True)) # tercera LSTM layer
regressor.add(Dropout(rate=0.2)) # tercer dropout

regressor.add(LSTM(units=50)) # cuarta LSTM layer
regressor.add(Dropout(rate=0.2)) # cuarto dropout

regressor.add(Dense(units=1)) # output layer

# Compilando la RNN
regressor.compile(optimizer='adam', loss='mean_squared_error') # MSE en vez de binary-crossentropy ya que es regresion

# Entrenando la RNN
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Guardando el modelo
regressor.save('Part 8 - Deep Learning/Supervised/3. Recurrent Neural Networks/rnn.hdf5')

# Cargando el modelo
from keras.models import load_model

regressor = load_model('Part 8 - Deep Learning/Supervised/3. Recurrent Neural Networks/rnn.hdf5')

# Importando el test set
dataset_test = pd.read_csv('Part 8 - Deep Learning/Supervised/3. Recurrent Neural Networks/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values # precios reales de las acciones en el primer mes de 2017

# Para predecir la tendencia de cada dia, necesitaremos los precios de los 60 dias anteriores, por eso, para predecir
# el primer mes de 2017, iremos necesitando tanto datos del training set, como del test set, ya que necesitaremos de
# 2016 y tambien de 2017. Para eso, los vamos a concatenar.
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1) # para que quede como un numpy array con una columna
inputs = scaler.transform(inputs)

X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60: i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prediciendo los valores de las acciones
pred_stock_price = regressor.predict(X_test)
pred_stock_price = scaler.inverse_transform(pred_stock_price)

# Visualizando los resultados
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(pred_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction with Recurrent Neural Network')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()