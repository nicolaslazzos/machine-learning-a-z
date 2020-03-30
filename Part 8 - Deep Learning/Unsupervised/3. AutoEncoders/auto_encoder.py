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