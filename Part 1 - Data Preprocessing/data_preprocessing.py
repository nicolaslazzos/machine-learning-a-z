import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Part 1 - Data Preprocessing/Data.csv')

# separar las matriz de variables independientes de la variable dependiente
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ocupandonos de los valores faltantes
from sklearn.impute import SimpleImputer

# reemplaza los NaN por la media de la columna
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encodear las variables categoricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# se transforma cada categoria a un numero
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# separar el dataset un training set y un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling, o lo que es lo mismo que normalizacion
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# aca no hace falta el fit, porque ya se le hizo el fit a X antes
X_test = sc_X.transform(X_test)

# en este caso, no hace falta feature scaling a Y ya que es una variable categorica