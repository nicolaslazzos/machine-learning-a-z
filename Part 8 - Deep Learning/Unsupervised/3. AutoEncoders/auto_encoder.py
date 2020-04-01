# AUTOENCODERS

# Los autoencoders poseen una capa de entrada, una hidden layer y una capa de salida. Desde laa neuronas de entrada
# hacia las neuronas de la hidden layer, realiza un proceso de "encoding", y luego de aca hacia las neuronas de salida,
# un proceso de "decoding", en el que apunta a que las salidas sean iguales o equivalentes a las entradas. Por lo tanto
# no seria un model de aprendizaje no supervisado puro, sino que es mas como auto-supervisado, ya que compara las
# salidas con las entradas. Pueden ser usados para Feature Detection, sistemas de recomendacion muy potentes y para
# encodear datos representandolos de una manera mas reducida.

# Al recibir los datos, estos son encodeados a un vector z de menor dimension, usando una funcion de mapeo, generalmente
# una sigmoide o una tangente hiperbolica (funciones de activacion). El vector z = f(Wx + b), donde x es el vector de
# entrada, W el vector de pesos y b el bias. En la salida el vector z es decodeado en un vector y, de las mismas
# dimensiones que x, apuntando a replicarlo. Finalmente se calcula el error de reconstruccion, y el objetivo es
# minimizarlo. Para esto, se ejecuta una backpropagation actualizando los pesos segun ese error calculado.

# Existe la posibilidad de tener mas nodos en la hidden layer que en las capas de entrada y salida, pero en este caso,
# habria un problema, y es que, para replicar los datos de entrada en la salida, al haber mayor o igual cantidad de
# nodos en la hidden layer, el autoencoder podria crear una correspondencia de igual a igual entre ciertos nodos de
# entrada con ciertos nodos ocultos, para de esta forma obtener una salida identica a la entrada y dejando posibles
# nodos sin usar en la hidden layer. Para resolver esto, hay diferentes tipos de AutoEncoders.

# Tipos de AutoEncoders:
#   - Sparse AutoEncoders: introducen una forma de regularizacion (como dropout regularization) que lo que hace, es que
#                          le impide a la red, usar todos los nodos ocultos en cada iteracion. Es decir, va desactivando
#                          algunos de ellos en cada iteracion.
#   - Denoising AutoEncoders: al tomar las entradas, no las ingresan directo, sino que en su lugar ingresan una copia
#                             de los valores de entrada, y reemplazando aleatoriamente algunos de ellos por cero, pero
#                             luego de calcular la salida, para el calculo del error, no se la compara con la entrada
#                             modificada, sino con la entrada original, por lo que entonces, no puede simplemente copiar
#                             los valores de entrada en la salida.
#   - Contractive AutoEncoders: introduce una penalizacion en la loss function que evita que se puedan simplemente
#                               copiar los valores.
#   - Stacked AutoEncoders: tienen mas nodos en las hidden layer pero divididos en dos o mas hidden layers. En este caso
#                           hay tantas fases de encoding como cantidad de hidden layers.
#   - Deep AutoEncoders: consiste en RBM apiladas.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn # modulo para nerual networks
import torch.nn.parallel # modulo para procesamiento paralelo
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable # para el stochastic gradient descend

# Importando el dataset
movies = pd.read_csv('Part 8 - Deep Learning/Unsupervised/2. Boltzmann Machines/ml-1m/movies.dat',
                     sep='::',
                     header=None,
                     engine='python',
                     encoding='latin-1')

users = pd.read_csv('Part 8 - Deep Learning/Unsupervised/2. Boltzmann Machines/ml-1m/users.dat',
                     sep='::',
                     header=None,
                     engine='python',
                     encoding='latin-1')

ratings = pd.read_csv('Part 8 - Deep Learning/Unsupervised/2. Boltzmann Machines/ml-1m/ratings.dat',
                     sep='::',
                     header=None,
                     engine='python',
                     encoding='latin-1')

# Preparando el training set y el test set
training_set = pd.read_csv('Part 8 - Deep Learning/Unsupervised/2. Boltzmann Machines/ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('Part 8 - Deep Learning/Unsupervised/2. Boltzmann Machines/ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Numero total de usuarios y peliculas
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convirtiendo los datos a una matriz con filas = usuarios, columnas = peliculas e interseccion = rating
def convert(data):
    new_data = []

    for user_id in range(1, nb_users + 1):
        movies_id = data[:, 1][data[:, 0] == user_id]
        ratings_id = data[:, 2][data[:, 0] == user_id]
        ratings = np.zeros(nb_movies)
        ratings[movies_id - 1] = ratings_id
        new_data.append(list(ratings))

    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convirtiendo los sets en Torch tensors (matrices multidimensionales de un solo tipo de datos, en este caso, Float)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Implementando el sistema de recomendacion para predecir la calificacion de peliculas con un Stacked Autoencoder

class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) # full connection entre la capa de entrada y la primer hidden layer (encoding)
        self.fc2 = nn.Linear(20, 10) # full connection entre la primer hidden layer y la segunda hidden layer (encoding)
        self.fc3 = nn.Linear(10, 20) # full connection entre la segunda hidden layer y la tercer hidden layer (decoding)
        self.fc4 = nn.Linear(20, nb_movies) # full connection entre la tercer hidden layer y la capa de salida (decoding)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Esta funcion realiza la forward propagation, es decir, pasa los datos de capa en capa realizando encoding y
        # decoding cuando corresponda.

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        return x

# Inicializando el Autoencoder
sae = SAE()
criterion = nn.MSELoss()

# El primer parametro son todos los parametros que definen la arquitectura de nuestro autoencoder
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Entrenando el Autoencoder
nb_epoch = 200

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # contador para normalizar el loss y calcular el MSE

    for user_id in range(nb_users):
        # Con esta linea se agrega una dimension a la input, ya que asi lo espera pytorch
        input = Variable(training_set[user_id]).unsqueeze(0)
        target = input.clone() # copia de las entradas para luego realizar la comparacion

        if torch.sum(target.data > 0): # se verifica si el usuario califico al menos una pelicula, sino no se procesa
            output = sae.forward(input)
            target.require_grad = False # para que el gradiente no sea calculado respecto del target, solo de la input
            output[target == 0] = 0 # volvemos a 0 los valores de peliculas que no habian sido calificadas

            loss = criterion(output, target)
            # Total de peliculas sobre cantidad de peliculas calificadas, se le agrega una cte para evitar que sea 0 el
            # denominador. Representa el error medio, pero considerando solo las peliculas que fueron calificadas.
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward() # indica en que sentido debe ser la correccion de los pesos (incrementar o reducir)
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.

            optimizer.step() # indica la cantidad en la que se deben ajustar los pesos, segun el sentido indicado antes

    print('Epoch: ' + str(epoch) + ' - Loss: ' + str(train_loss/s)) # da un error de aproximadamente 1 sobre 5 estrellas

# Testeando el Autoencoder
test_loss = 0
s = 0. # contador para normalizar el loss y calcular el MSE

for user_id in range(nb_users):
    # En este caso, como estamos testeando, el objetivo es predecir el rating de las peliculas del test set, y es
    # por eso, que como entrada, se continua usando el training set, ya que a partir de lo que la red aprenda de los
    # datos que conoce, buscara predecir aquellos que no conoce, o sea, el test set.
    input = Variable(training_set[user_id]).unsqueeze(0)
    target = Variable(test_set[user_id]).unsqueeze(0)

    if torch.sum(target.data > 0): # se verifica si el usuario califico al menos una pelicula, sino no se procesa
        output = sae.forward(input)
        target.require_grad = False # para que el gradiente no sea calculado respecto del target, solo de la input
        output[target == 0] = 0 # volvemos a 0 los valores de peliculas que no habian sido calificadas

        loss = criterion(output, target)
        # Total de peliculas sobre cantidad de peliculas calificadas, se le agrega una cte para evitar que sea 0 el
        # denominador. Representa el error medio, pero considerando solo las peliculas que fueron calificadas.
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.

print('Test Loss: ' + str(test_loss/s)) # da un error de aproximadamente 1 sobre 5 estrellas