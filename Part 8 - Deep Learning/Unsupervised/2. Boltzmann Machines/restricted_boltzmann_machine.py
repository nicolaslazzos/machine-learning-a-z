# BOLTZMANN MACHINES (BM)

# Todos los tipos de redes vistos anteriormente tienen en comun que son modelos dirigidos, es decir, hay una direcccion
# en la que el modelo funciona. En las Boltzmann Machines, no hay direccionalidad, son como un grafo conexo completo y
# no dirigido.

# Las BM, no tienen una capa de salida. Tiene nodos visibles y nodos ocultos, pero no discrimina nodos, sino que los
# trata a todos de la misma forma. No tiene una capa de entrada debido a que no recibe informacion, sino que la genera.
# Si lo comparamos con una planta nuclear por ejemplo, tiene muchas partes, de las cuales algunas se miden y otras no, y
# entre todas, forman un unico sistema. En una BM los nodos visibles representan cosas que medimos o podemos medir,
# mientras que los nodos ocultos son cosas que no medimos o no podemos medir. No se quedan esperando entradas, ya que
# son capaces de generar por si mismas un conjunto de diferentes estados, variando cada uno de los parametros que cada
# nodo representa. En el caso de compararlo con la planta nuclear, podria por ejemplo combinar distintas temperaturas
# de ambiente, con distintas velocidades de viento y distintas presiones atmosfericas, etc. Por esto, no es un modelo
# determinista, sino estocastico.

# Lo que se hace, es entrenarla con datos para ayudarla a ajustar los distintos pesos del sistema acorde a los mismos y
# de esta manera aprende los distintos valores que puede tomar cada parametro o caracteristica, como se combinan entre
# ellos o como pueden afectarse, como interactuan, cuales son las conexiones y de esta forma nos permitiria, por ejemplo
# en el caso de una planta nuclar, monitorearla, ya que la BM estaria modelando la planta nuclear. Del mismo modo, nos
# permitiria identificar y recrear estados anormales de funcionamiento.

# Energy-Based Models (EBM)

# Las BM reciben este nombre debido a que usan la distribucion de Boltzmann para crear los diferentes estados. La
# formula de Boltzmann para cada estado representa la probabilidad de ocurrencia del mismo, y dicha probabilidad es
# inversamente proporcional a la cantidad de energia del sistema, es decir, mientras mas energia se necesita, mas
# dificil es que el estado ocurra.

# En las BM, la energia esta representada por los diferentes pesos de las sinapsis, por lo tanto, una vez entrenada la
# red, buscara siempre el estado de menor energia.

# Restricted Boltzmann Machines (RBM)

# Es el mismo concepto, pero con la diferencia que los nodos visibles no se conectan entre ellos, al igual que los nodos
# ocultos. Esto debido a que cuando el numero de neuronas aumenta, se vuelve muy costoso computar todas las conexiones
# en una BM tradicional.

# Durante el proceso de aprendizaje, lo que la RBM va a aprender, es como asignar sus nodos ocultos, a ciertas features
# o variables de los datos que se estan analizando. Por ejemplo, en un sistema de recomendacion de peliculas, podria
# aprender que ciertos generos son importantes, y ciertos actores y directores, por lo que los iria asignando a ciertos
# nodos ocultos. No necesariamente deben ser esas features, solo son un ejemplo para nuestro entendimiento. Los nodos
# ocultos se activaran segun si dicha feature esta presente en las peliculas que al usuario le han gustado. En el caso
# contrario, no se encenderan. Por ejemplo, si un nodo representa el genero Accion, y al usuario le gustaron peliculas
# de dicho genero, el mismo se encendera. Al ingresar una nueva pelicula, para saber si la recomienda o no, evaluara
# sus conexiones con los nodos ocultos, en funcion de los pesos de las sinapsis, y si los nodos estan activados o no.

# Contrastive Divergence

# Es la forma en las que las RBM ajustan sus pesos. A partir de pesos inicializados aleatoriamente, la RBM calcula los
# nodos ocultos, y luego, estos son capaces de reconstruir o recalcular las entradas o los nodos visibles usando estos
# pesos. Cabe aclarar, que los valores resultantes no seran exactamente igual a los originales, ya que los nodos
# visibles, no estan interconectados entre ellos, pero al ser recalculados por los nodos ocultos, estos estan ultimos
# recibiendo informacion de todos los nodos visibles, por lo que el recalculo no sera exactamente el mismo. Este
# proceso se repite por cada nueva entrada, es decir, se ingresan nuevos valores a los nodos visibles, se calculan los
# nodos ocultos, y estos recalculan los nodos visibles. Termina cuando los nodos visibles recalculados no cambian, es
# decir, converge.

# Durante este proceso, el sistema al ajustar sus pesos, siempre tratara de buscar el estado de menor energia, y los
# valores que obtendremos al final, en los nodos recalculados, no seran los que realmente buscamos. Entonces, lo que se
# busca con este proceso es ver en que sentido va la "curva de energia" a medida que se entrena la red, para luego
# modificar los pesos iniciales, de tal forma que al ingresar los datos de entrada o de entrenamiento (que son los
# valores reales de los nodos visibles), la red se encuentre en su estado de menor energia.
# (Ver video "Contrastive Divergence")

# Sin embargo no hace falta realizar el proceso completo hasta la convergencia, sino que con unos primeros pasos ya es
# suficiente para ver en que sentido va la "curva de energia" y ajusta los pesos iniciales.

# Deep Belief Netowrks (DBN)

# Surgen de apilar dos o mas RBM, y en este caso, excepto por la ultima capa, las demas estan dirigidas hacia abajo,
# luego de ser entrenadas. Se entrena capa por capa.

# Deep Boltzmann Machines (DBM)

# Son como las anteriores, excepto que no direcciona las capas, sino que las deja todas sin dirigir. Ademas se especula
# que las BBM pueden extraer features que son mas sofisticadas o complejas, y por lo tanto podrian ser usadas para
# tareas mas complejas.

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

# Implementando el sistema de recomendacion con RBM

# Convirtiendo los ratings en binarios (Liked = 1, Not Liked = 0 y Sin Calificar = -1)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creando la arquitectura de la red neuronal
class RBM():
    def __init__(self, nv, nh): # nv = cantidad de nodos visibles, nh = cantidad de nodos ocultos
        self.W = torch.randn(nh, nv) # inicializacion aleatoria de los pesos (vector de pesos)
        self.a = torch.randn(1, nh) # bias (probabilidad de los nodos ocultos dado los nodos visibles)
        self.b = torch.randn(1, nv) # bias (probabilidad de los nodos visibles dado los nodos ocultos)

        # El bias permite mover la funcion de activacion sobre el eje x para ajustarse mejor a las entradas. Es decir,
        # es basicamente otro peso. En este caso, con a y b se inicializan aleatoriamente lo bias para cada nodo.

    def sample_h(self, x):
        # Esta funcion va a tomar muestras de las activaciones de todos los nodos ocultos de la red. Para esto, los va a
        # activar, de acuerdo a una probabilidad p(h, v), es decir, la probabilidad de que el nodo h sea 1, dado el
        # valor de v. Esta probabilidad es igual a la funcion de activacion (la funcion de activacion sigmoid) y la
        # calcularemos en esta funcion. El parametro x, es el vector de valores de los nodos visibles (v en la formula).

        WX = torch.mm(x, self.W.t()) # se toma la transpuesta de W debido a que v se corresponde con las columnas
        activation = WX + self.a.expand_as(WX) # neuronas * pesos + bias
        p_h_given_v = torch.sigmoid(activation)

        # El primer parametro que se retorna, es la probabilidad de activacion de cada nodo oculto datos los valores de
        # los nodos visibles. Luego, el segundo parametro, es el muestreo de activacion de cada nodo oculto, es decir,
        # para cada uno se genera un numero aleatorio entre 0 y 1, y en funcion de su probabilidad de activacion, se
        # determina si el nodo se activaria o no.

        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        # Esta funcion hara lo mismo que sample_v, pero para los nodos visibles. Hay un nodo visible por cada pelicula
        # por lo que el vector de probabilidades tendra la probabilidad de que la pelicula le guste o no al usuario,
        # calculada en funcion de los valores de los nodos ocultos.

        WY = torch.mm(y, self.W) # neuronas * pesos
        activation = WY + self.b.expand_as(WY) # neuronas * pesos + bias
        p_v_given_h = torch.sigmoid(activation)

        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        # Esta funcion implementa la Contrastive Divergence. Lo que hace es, samplear los nodos ocultos a partir de los
        # valores de los nodos visibles. Luego con estos valores, samplea los nodos visibles y obtiene nuevos valores.
        # Y con estos valores, vuelve a samplear los nodos ocultos y obtiene nuevos valores, y asi sucesivamente.

        # v0 = valores iniciales de los nodos visibles (ratings de un usuario para cada pelicula)
        # vk = valores de los nodos visibles obtenidos luego de k muestreos
        # ph0 = las probabilidades de los nodos ocultos, durante el primer muestreo, dados los valores de los v0
        # phk = las probabilidades de los nodos ocultos, luego de k muestreos, en funcion de los valores de los vk

        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        # self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Inicializando la RBM
nv = len(training_set[0]) # un nodo visible por cada pelicula
nh = 100 # el numero de nodos ocultos se elige, estimativamente, en funcion de la cantidad de nodos visibles
batch_size = 100

rbm = RBM(nv, nh)

# Entrenando la RBM
nb_epoch = 10

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # contador para normalizar el loss

    for user_id in range(0, nb_users - batch_size, batch_size): # se va tomando un batch de 100 usuarios
        vk = training_set[user_id: user_id + batch_size]
        v0 = training_set[user_id: user_id + batch_size]
        ph0, _ = rbm.sample_h(v0)

        for k in range(10): # Gibbs sampling
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0] # volvemos los nodos de las pelis no calificadas a -1 para que la red no aprenda de ellos

        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, hk)

        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # Average Distance
        # train_loss += np.sqrt(torch.mean((v0[v0 >= 0] - vk[v0 >= 0]) ** 2)) # RMSE
        s += 1.

    print('Epoch: ' + str(epoch) + ' - Loss: ' + str(train_loss/s)) # dio una perdida del 25% aproximadamente

# Testeando la RBM
test_loss = 0
s = 0. # contador para normalizar el loss

for user_id in range(nb_users): # se va tomando un batch de 100 usuarios
    # En este caso, como estamos testeando, el objetivo (vt) es predecir el reting de las peliculas del test set, y es
    # por eso, que como entrada, se continua usando el training set, ya que a partir de lo que la red aprenda de los
    # datos que conoce, buscara predecir aquellos que no conoce, o sea, el test set.

    v = training_set[user_id: user_id + 1]
    vt = test_set[user_id: user_id + 1]

    if len(vt[vt >= 0]) > 0: # si hay al menos una pelicula calificada, se pueden hacer predicciones
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)

        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])) # Average Distance
        # train_loss += np.sqrt(torch.mean((vt[vt >= 0] - v[vt >= 0]) ** 2)) # RMSE
        s += 1.

print('Test Loss: ' + str(test_loss/s)) # dio una perdida similar al training, lo que es muy bueno