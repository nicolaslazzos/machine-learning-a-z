# THOMPSON SAMPLING

# El problema es cuestion es el mismo que se analizo en el UCB (Multi-Armed Bandit aplicado a un caso de publicidades)

# El Thompson sampling va probando varias veces cada publicidad y asi va formando como una especie de distribucion de
# probabilidad de la efectividad de cada publicidad, es decir, del valor de retorno, dado por si la gente ha clickeado
# o no cada una cuando se le presento. Luego, lo que hace es a partir de cada distribucion, generar un valor aleatorio
# y elige para mostrar la publicidad correspondiente a la distribucion que genero el mayor valor.
# Luego de mostrarle esa publicidad al usuario, el usuario clickeara o no, y en base a esa informacion, se actualiza
# la distribucion de la publicidad correspondiente y se elige nuevamente una publicidad a mostrar de la misma forma que
# se hizo antes.

# A diferencia del UCB, que es deterministico, el Thompson Sampling, es probabilistico. Ademas, el UCB necesita
# actualizar sus valores si o si luego de cada ronda para definir que ocurre en la siguiente, en cambio el Thompson
# Sampling, no necesariamente deben actualizarse despues de cada ronda, sino que los resultados de cada ronda pueden
# irse acumulando y cada cierta cantidad de rondas, se actualiza el algoritmo con toda la informacion recopilada.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CTR = Click-Trhough Rate
dataset = pd.read_csv('Part 6 - Reinforcement Learning/1. Upper Confidence Bound/Ads_CTR_Optimisation.csv')

# El dataset contiene 10 publicidades distintas con 10mil usuarios, en los que para cada uno, indica cual o cuales
# publicidades de las 10, va a clickear. Esto va a servir para simular usuarios, que van a clickear o no en funcion
# de la publicidad que se le muestre, la cual dependera de los resultados obtenidos en rondas anteriores.

# Implementando Thompson Sampling
import random

N = 10000 # cantidad de rondas
d = 10 # cantidad de publicidades
ads_selected = []
number_of_selections = [0] * d
sums_of_rewards = [0] * d

for n in range(0, N):
    max_random = 0
    ad = 0

    for i in range(0, d):
        random_beta = random.betavariate((sums_of_rewards[i] + 1), (number_of_selections[i] - sums_of_rewards[i] + 1))

        if random_beta > max_random:
            max_random = random_beta
            ad = i

    ads_selected.append(ad)
    number_of_selections[ad] += 1
    sums_of_rewards[ad] += dataset.iloc[n, ad]

total_reward = sum(sums_of_rewards)

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of Selections')
plt.show()