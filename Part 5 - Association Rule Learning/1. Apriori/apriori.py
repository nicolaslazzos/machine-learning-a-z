# APRIORI

# Gente que compro tal cosa, tambien compro tal otra cosa...

# El algoritmo Apriori tiene 3 partes:
#   - Support(X) = (gente que compro X producto) / (total de transacciones)
#                  Es la proporcion de transacciones en la que el producto X aparece. Representa su popularidad.
#   - Confidence(X --> Y) = (gente que compro el producto X y el producto Y) / (gente que compro el producto X) -->
#                  (support(X) U support(Y)) / support(X)
#                  aca estamos probando una hipotesis, de que los que compran el producto X, tambien compran el producto
#                  Y. Mientras mas cerca de 1 de el resultado, mas fuerte es la hipotesis.
#                  Representa la probabilidad de que el producto Y sea comprado cuando se compra el producto X.
#                  Representa cual es la probabilidad de que al recomendar la pelicula X a una nueva persona que ha
#                  visto la pelicula Y, le guste.
#   - Lift(X --> Y) = confidence / support(Y) --> (support(X) U support(Y)) / (support(X) * support(Y))
#                     Es la probabilidad de que el producto Y sea comprado cuando se compra el producto X, pero teniendo
#                     en cuenta aquellos que han comprado Y.

# Al calcular el Support, estamos calculando la popularidad de cierto producto X. Si nos da por ejemplo 10%, entonces
# sabemos que al ofrecerle ese producto a un nuevo individuo, hay un 10% de probabilidades de que lo compre. Ahora,
# cuando calculamos la Confidence, calculamos la probabilidad de que una persona que compro el producto X, compre el
# producto Y. Por lo tanto, luego al calcular el Lift, al estar sumandole al Support de X, informacion que nos brinda
# la Confidence, ahora podemos en vez de ofrecerle el producto X a cualquier persona, ofrecerselos a aquellos que han
# comprado tambien el producto Y, por lo tanto, tenemos mas probabilidad de que un nuevo individuo lo compre,
# y es justamente esa mejora lo que representa el Lift.

# Pasos del Algoritmo:
#   1. Establecer un Support y Confidence minimos (un limite inferior).
#   2. Tomar todos los items que poseen un Support mayor al limite
#   3. Tomar todas las reglas de asociacion que tengan una Confidence mayor al limite
#   4, Ordenar las reglas en funcion del Lift y de forma descendente

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset con transacciones de una tienda en francia
dataset = pd.read_csv('Part 5 - Association Rule Learning/1. Apriori/Market_Basket_Optimisation.csv', header=None)

transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])])

# entrenando el algoritmo Apriori con el dataset
from apyori import apriori

# para el min_support estamos considerando un producto que se compra al menos 3 veces por dia
# en el nivel de confianza minimo, se debe evitar fijar uno muy alto, ya que esto puede llevar a que se creen reglas
# entre productos que son comprados en conjunto no porque tengan alguna relacion, sino porque son productos muy
# comprados en general
# min_length indica el largo minimo de la regla (se deja en 2 para evitar reglas de un solo item)
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# visualizando los resultados
results = list(rules)

# los resultados se obtienen ordenados por un criterio de reelevancia basado en los 3 parametros del algoritmo
