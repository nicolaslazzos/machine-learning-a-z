# LOGISTIC REGRESSION

# Formula --> Ln (p/(1-p)) = b0 + (b1 * x)

# En este caso, como hablamos de clasificacion, los valores a predecir, son categorias, valores discretos, no
# continuos como en la regresion lineal, por lo tanto, no puede usarse la linea de regresion para predecir las
# categorias. Por eso, a la linea de regresion, se le aplica una funcion Sigmoide, con lo que obtenemos una nueva
# funcion, que no va a predecir las categorias en si, sino la probabilidad de una muestra de pertenecer a una
# categoria o a otra.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 3 - Classification/1. Logistic Regression/Social_Network_Ads.csv')

# separar las matriz de variables independientes de la variable dependiente

# matriz de variables independientes (Gender and Age)
X = dataset.iloc[:, 2:-1].values

# vector de variables dependientes
y = dataset.iloc[:, -1].values

# separar el dataset un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling, o lo que es lo mismo que normalizacion
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# aca no hace falta el fit, porque ya se le hizo el fit a X antes
X_test = sc_X.transform(X_test)

# creando y entrenando el modelo
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# (0;0) predicciones de la clase 0 correctas
# (1;0) predicciones de la clase 0 incorrectas (eran de la clase 1)
# (0;1) predicciones de la clase 1 incorrectas (eran de la clase 0)
# (1;1) predicciones de la clase 1 correctas

# grafica de la clasificacion (regiones de prediccion)
# al ser un modelo de clasificacion lineal, el limite de prediccion es una linea recta
from matplotlib.colors import ListedColormap

def plot_classification(X_set, y_set, title):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    colors = ('red', 'green')
    plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(colors))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for cat, color in zip(np.unique(y_set), colors):
        plt.scatter(X_set[y_set == cat, 0], X_set[y_set == cat, 1], c=color, label=cat)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

# Training set
plot_classification(X_train, y_train, 'Logistic Regression (Training Set)')

# Test set
plot_classification(X_test, y_test, 'Logistic Regression (Test Set)')