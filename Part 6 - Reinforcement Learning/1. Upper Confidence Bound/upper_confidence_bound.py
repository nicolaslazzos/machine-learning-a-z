# UPPER CONFIDENCE BOUND (UCB)

# Tenemos el problema del Multi-Armed Bandit, que en este caso, consta de diferentes publicidades que van a ser
# mostradas a los usuarios, y se desea saber cual de todas ellas es la mejor, para optimizar nuestras campaña
# publicitaria. Para ello, tenemos D publicidades, que van a ser mostradas a los usuarios cada vez que se conectan
# a una pagina web (solo se le puede mostrar una por vez). Si el usuario clickea en la publicidad, la misma sera
# recompensada (1), de lo contrario, si no clickea, no sera recompensada (0). El objetivo es maximizar la recompensa
# total luego de varias rondas.

# Pasos del algoritmo:
#   1. Durante cada ronda, se consideran dos valores:
#       Ni(n): cantidad de veces que la publicidad i ha sido mostrada durante las n rondas
#       Ri(n): la suma de recompensas de la publicidad i durante las n rondas
#   2. A partir de los dos valores anteriores se calcula:
#       - La recompensa promedio de la publicidad i durante las n rondas
#       - El invtervalo de confianza (Confidence Bound) en la ronda n
#   3. Se selecciona la publicidad i que tiene el maximo Upper Confidence Bound

# Inicialmente, el algoritmo define un valor de recompensa y un Confidence Bound inicial e igual para todas las
# publicidades. Luego, elige aquella con el mayor Confidence Bound y se la muestra al usuario. Si el usuario por
# ejemplo no la clickea, entonces el valor de recompensa promedio para esa publicidad desciende y a continuacion se
# reduce el Confidence Bound. Luego se vuelve a elegir aquella con el mayor Confidence Bound, y si en este caso el
# usuario la clickea, entonces su valor de recompensa promedio aumenta y se reduce su Confidence Bound, ya que al tener
# ya una observacion, tenemos mas confianza de su efectividad. Este proceso continua asi, hasta que el valor de
# recompensa promedio converge a un cierto valor y el Confidence Bound de alguna de ellas (la mejor) se ha reducido
# casi por completo. Ese valor de recompensa promedio, representa algo asi como la efectividad de esa publicidad.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CTR = Click-Trhough Rate
dataset = pd.read_csv('Part 6 - Reinforcement Learning/1. Upper Confidence Bound/Ads_CTR_Optimisation.csv')

# El dataset contiene 10 publicidades distintas con 10mil usuarios, en los que para cada uno, indica cual o cuales
# publicidades de las 10, va a clickear. Esto va a servir para simular usuarios, que van a clickear o no en funcion
# de la publicidad que se le muestre, la cual dependera de los resultados obtenidos en rondas anteriores.

# Implementando UCB
import math

N = 10000 # cantidad de rondas
d = 10 # cantidad de publicidades
ads_selected = []
number_of_selections = [0] * d
sums_of_rewards = [0] * d

for n in range(0, N):
    max_upper_bound = 0
    ad = 0

    for i in range(0, d):
        # en las primeras 10 rondas, como no hay datos iniciales, se va a seleccionar cada publicidad una vez
        if number_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            # si esta publicidad aun no ha sido elegida, se setea un upper_bound muy grande para que sea la elegida
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
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