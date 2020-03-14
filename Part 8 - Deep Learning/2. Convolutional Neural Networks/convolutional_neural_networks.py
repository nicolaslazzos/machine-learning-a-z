# CONVOLUTIONAL NEURAL NETWORKS

# Recibe por entrada una imagen que es procesada por la red convolucional y retorna como salida una etiqueta

# Para una imagen en blanco y negro, la misma se representa como un array 2d en en que cada pixel tiene asociado un
# valor entre 0 y 256 dependiendo su intensidad (0 =  negro y 256 = blanco).
# En el caso de una imagen a color, se representa con un array 3d, correspondientes a los colores primarios (red, green,
# blue), en donde para cada color, se asocia al pixel la intensidad al igual que en la imagen en blanco y negro, con un
# valor entre 0 y 256.

# Luego, los pasos a aplicar son: Convolution --> Max Pooling --> Flattering --> Full Connection

# Paso 1: Convolucion

# Se define una matriz de n x n pixeles, lamada Feature Detector o Filtro que se aplica a la imagen pasandolo de
# izquierda a derecha y de arriba hacia abajo, multiplicando cada pixel del filtro con el pixel de la imagen
# correspondiente a la misma posicion. Como resultado se obtiene un Activation o Feature Map. Como resultado, una de las
# cosas, es que la imagen resulta de menor tamaño, esto dependiendo el tamaño del Filtro o Feature Detector. Mientras
# mas coincidan los valores de los pixeles del Feature Map con los pixeles, mayor sera el valor del pixel resultado
# en el Feature Map. Los valores altos indican las coincidencias.

# Lo que se hace con eso, es buscar coincidencias en ciertas formas, lo que es equivalente a lo que los humanos hacemos
# al ver una imagen, es decir, no analizamos pixel por pixel, sino que reconocemos formas, como por ejemplo, al ver una
# imagen de una persona, podemos reconocer los ojos, la nariz, la boca, etc. Eso es lo que busca el Feature Detector al
# aplicarlo a una imagen, reconocer formas, y los resultados los obtenemos en el Feature Map como valores numericos.

# Es por eso, que se crean varios Feature Maps, con distintos Filtros o Feature Detectors (diferentes formas) que la red
# decide a traves de su entrenamiento, es decir, decide que formas o Features son importantes para ciertas categorias o
# ciertos tipos y entonces las busca. El conjunto de Feature Maps, crea la primer Convolutional Layer.

# Luego a la Concolutional Layer, se le aplica la Rectufier Linear Unit Function, para aumentar la no linearidad, ya que
# las imagenes son altamente lineales, en especial si contienen muchos objetos o detalles.

# Paso 2: Max Pooling

# La red debe tener Spacial Invariance, es decir, no debe importar en donde estan situadas las Features de cada imagen,
# ya que al analizar imagenes de personas, estas pueden estar en distintas posiciones o perspectivas, o el fondo puede
# ser distinto, o la luminosidad, o el lugar de la imagen en el que esta situado la persona, o el angulo, etc.

# Entonces lo que hace el Max Pooling, es recorrer cada Feature Map con una matriz de n x n pixeles, de izquierda a
# derecha y de arriba hacia abajo, sin pisarse, y va tomando el mayor valor que abarca la matriz en cada paso. Como
# resultado se obtiene un Poolead Feature Map, que al estar tomando un valor en particular por cada area del Feature Map
# sera igual o similar sin importar si en una de las imagenes la Feature estaba un poco hacia el costado o un poco
# rotada. A su vez, otro beneficio es que nuevamente se esta reduciendo el tamaño de la imagen, es decir, el numero de
# parametros que iran a la red neuronal, lo que ayuda a prevenir el overfitting.

# Hay mas tipos de Pooling ademas del Max Pooling.

# Paso 3: Flattening

# Se toma los valores de cada pixel fila por fila y se los coloca en un array 1d, ya que esto sera la entrada de una
# futura ANN que realizara el proceso de clasificacion.

# Paso 4: Full Connection

# Se agrega una red neuronal (ANN) Fully Connected (las Hidden Layers estan conectadas con todos los nodos, cosa que
# en una ANN no necesariamente debe ser asi), que recibira como entrada el vector que se obtuvo en el paso anterior. Lo
# que hara la ANN es combinar estos atributos de entrada para generar nuevos atributos, para incluso mejorar la
# efectividad del algoritmo. Finalmente se obtendran tantas posibles salidas como categorias haya, y una de ellas se
# activara dependiendo de la entrada y la prediccion.

# En la Backpropagation, ademas de ajustarse los pesos de la ANN, tambien se ajustaran los Feature Detectors.

# Al activarse ciertas neuronas, la red aprendera cuales neuronas se activan cuando la entrada es cierta categoria y
# cuales no se activan. Tambien sabra que cuando dicho conjunto de neuronas se activan, no son significativas para otras
# ciertas categorias, ya que para estas otras, se activan otros conjuntos de neuronas diferentes. Es decir, las neuronas
# de salida, dependiendo la categoria a la que pretenecen, aprenderan que neuronas de la capa anterior deben activarse
# para que ellas se activen.

# Softmax & Cross-Entropy

# La softmax function se aplica a las neuronas de salida y es un tipo de normalizacion, que asegura que la suma de las
# salidas es igual a 1. La Cross-Entropy por su lado, es la funcion que se aplica para calcular el costo, en este caso
# llamado Loss, que luego le sirve a la red para ajustar los pesos. Para mejorar la performance de la red, el objetivo
# es minimizar el Loss. Se usa esta funcion en lugar de por ejemplo el MSE, ya que al poseer un logaritmo, al mejorar
# la red, el cambio o la mejora, se ve mucho mas reflajada en el resultado, debido a que hay un mayor cambio. Por esto,
# en el Backpropagation, el feedback que se obtiene es mayor. Por el otro lado, en el MSE, este cambio no se nota mucho
# por mas que la red haya mejorado bastante. Cabe aclarar que el Cross-Entropy es la funcion favorita para estas redes
# pero unicamente en el caso de una clasificacion.

