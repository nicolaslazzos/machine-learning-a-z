# NATURAL LANGUAGE PROCESSING (NLP)

# El foco principal del NLP, es enseñar a las maquinas que es lo que dice en un texto ecrito o hablado.

# Principales librerias de NLP
#   - Natural Language Toolkit (NLTK)
#   - SapCy
#   - Stanford NLP
#   - OpenNLP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Vamos a analizar un conjunto de reviews de un restaurante e identificar en base al texto, si la misma es positiva o no
dataset = pd.read_csv('Part 7 - Natural Language Processing/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Primero se limpian los textos, esto significa por ejemplo sacar conectores, la puntuacion, y palabras que no son
# reelevantes. Tambien se eliminan los numeros, a menos que puedan tener un impacto significativo. Tambien se aplicara
# Stemming, que significa tomar la palabra raiz de un conjunto de diferentes versiones de una misma palabra, como por
# ejemplo, las conjugaciones de una misma palabra. El objetivo de esto, es eliminar palabras redundantes, ya que luego
# para cada palabra habra una columna en la sparse matrix, asi que mientras menos haya, mejor .Por ultimo se eliminaran
# las mayusculas.
# Luego de esto, lo que se hara entonces es separar cada review en un conjunto de palabras, que gracias al
# preprocesamiento, seran palabras reelevantes. Luego se tomaran las palabras de todas las reviews, y se asignara una
# columna a cada palabra y una fila a cada review, por lo tanto, cada fila x columna, tendra un numero indicando
# cuantas veces cada palabra aparece en cada review.

# limpiando los textos
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# corpus = coleccion de textos
corpus = []
ps = PorterStemmer()

for review in dataset.Review:
    review = re.sub('[^a-zA-Z]', ' ', review).lower().split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creando el modelo Bag of Words. Es basicamente crear la matriz en donde cada columna es una palabra y cada fila es una
# review, siendo la interseccion de una review, con una palabra, la cantidad de veces que esa palabra aparece en la
# review (la llamada Sparse matrix). Esto debe hacerse, ya que el modelo, con el fin de clasificar las reviews, primero
# debe ser entrenado sobre el conjunto de reviews, y para esto, necesita variables independientes (las palabras de cada
# review) y la variable dependiente (si es positiva o negativa dicha la review)
from sklearn.feature_extraction.text import CountVectorizer

# max_freatures es para limitar el numero de palabras y asi evitar las que son poco relevantes y aparecen pocas veces
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.Liked.values

# Los modelos de clasificacion mas comunes en NLP suelen ser Naive Bayes o Decision Tree o Random Forest Classification
# Otros tambien pueden ser CART, C5.0 y Maximum Entropy

# Aplicando Naive Bayes

# separar el dataset un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# creando y entrenando el modelo
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# precision = 73% aprox.