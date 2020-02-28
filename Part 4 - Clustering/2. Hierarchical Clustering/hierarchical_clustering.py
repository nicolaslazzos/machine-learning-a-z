# HIERARCHICAL CLUSTERING (HC)

# Hay dos tipos de HC:
#   - Agglomerative: arranca desde abajo, va agrupando las muestras en clusters
#   - Divisive: arranca desde arriba, va dividiendo los datos en clusters

# Agglomerative

# Primero se arranca tomando cada punto o dato como un cluster individial. Despues se toman los dos puntos mas cercanos
# y se los agrupa en un cluster. Luego se toman los 2 clusters mas cercanos y se los agrupa en un unico cluster, y asi
# sucesivamente hasta que se tiene un unico cluster.
# Durante este procedimiento, el algoritmo va recordando los pasos que fue tomando y los refleja en un Dendrograma, que
# muestra como se fueron formando los clusters y las distancias entre ellos. Por lo tanto, como HC se ejecuta hasta
# formar un solo cluster, lo que podemos hacer es definir un limite de disimilaridad, que hara que se conserven aquellos
# clusters cuya distancia o disimilaridad este por debajo de ese limite. Una recomendacion para definir el limite es
# que corte la distancia mas grande (que no es cruzada por otra linea horizontal) en el Dendrograma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 4 - Clustering/1. K-Means Clustering/Mall_Customers.csv')

# matriz de variables independientes (Annual Income y Spending Score)
X = dataset.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as sch

# usando el Dendrograma para determinar el numero optimo de clusters (K)
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')

# numero optimo de clusters = 5

from sklearn.cluster import AgglomerativeClustering

# entrenando el modelo
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

# fit_predict entrena el modelo y retorna a que cluster pertenece cada punto
y_hc = hc.fit_predict(X)

# graficando los clusters
clusters_labels = np.unique(y_hc)
colors = ('red', 'lightblue', 'green', 'orange', 'purple')
clusters_names = ('Careful', 'Standard', 'Target', 'Careless', 'Sensible')

for cluster, color, label in zip(clusters_labels, colors, clusters_names):
    plt.scatter(X[y_hc == cluster, 0], X[y_hc == cluster, 1], c=color, label=label)

plt.legend()
plt.title('Mall Customers Clusters (Hierarchical Clustering)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# se obtuvo el mismo resultado que con K-Means