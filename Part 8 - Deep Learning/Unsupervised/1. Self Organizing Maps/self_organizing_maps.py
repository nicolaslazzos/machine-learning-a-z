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
# alejamos el BMU.

# Por cada epoch, el radio que abarca cada BMU, va reduciendose, de forma que al actualizarse los pesos, van teniendo
# impacto sobre una menor cantidad de puntos o neuronas a su alrededor. Asi el proceso va volviendose cada vez mas
# preciso, realizando ajustes mas pequeños o especificos. Al ser un tipo de red de aprendizaje no supervisado, no
# efectua backpropagation, ya que no tiene un objetivo con el cual comprar.

