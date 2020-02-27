# K-MEANS CLUSTERING

# El clustering busca agrupar los datos en grupos o clusters que no han sido definidos previamente, a diferencia de
# la clasificacion.

# En K-Means se debe elegir un numero K clusters a generar. Luego se seleccionan K centroides, alrededor de los cuales
# se acumularan las diferentes muestras (en funcion de su cercania) para formar esos clusters. Es decir, se asigna
# cada punto al grupo perteneciente al centroide mas cercano. El metodo mas comun para calcular estas distancias, es
# la distancia Euclidea. Una vez que se generaron los K clusters, se recalculan los centroides en funcion de los datos.
# Luego se vuelve a asignar cada punto o muestra al grupo perteneciente al centroide mas cercano, y asi sucesivamente
# hasta que no hay cambios en las asignaciones.

# La inicializacion aleatoria de los centroides puede llevar a un clustering erroneo o un resultado no muy bueno. Para
# evitar, esto, existe un metodo de inicializacion llamado KMeans++

# Para saber cuantos clusters seleccionar, se utiliza una metrica llamada WCSS (Within Clusters Sum of Squares), que
# calcula la suma del cuadrado de las distancias entre cada punto y el centroide del cluster al cual pertenece. A medida
# que aumentan los clusteres, decrece el WCSS, ya que se achican esas distancias al centroide. Por lo tanto, el WCSS
# siempre va a decrecer, hasta llegar a cero (nro de clusters = nro de muestras). Para saber cuando detenernos (cuando
# dejar de agregar clusteres) se debe observar a los cambios del WCSS, cuando el cambio ya no es muy significativo
# significa que probablemente no se debe continuar agregando clusteres. Esto puede observarse como un quiebre en un
# grafico de linea si graficamos los distintos WCSS a medida que aumentamos la cantidad de clusteres. Este metodo se
# conoce como The Elbow Method. (Es una metrica, pero no significa que sea la unica forma de decidir el nro de clusters)

# El WCSS tambien es conocido como Inertia

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Part 4 - Clustering/1. K-Means Clustering/Mall_Customers.csv')

# matriz de variables independientes (Annual Income y Spending Score)
X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans

# usando The Elbow Method para determinar el numero de clusters (K)
wcss = []
for i in range(1, 11): # probando de 1 a 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# numero optimo de clusters = 5

# aplicando K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

# fit_predict entrena el modelo y retorna a que cluster pertenece cada punto
y_kmeans = kmeans.fit_predict(X)

# graficando los clusters
clusters_labels = np.unique(y_kmeans)
colors = ('red', 'lightblue', 'green', 'orange', 'purple')
clusters_names = ('Careful', 'Standard', 'Target', 'Careless', 'Sensible')

for cluster, color, label in zip(clusters_labels, colors, clusters_names):
    plt.scatter(X[y_kmeans == cluster, 0], X[y_kmeans == cluster, 1], c=color, label=label)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=100, label='Centroids')

plt.legend()
plt.title('Mall Customers Clusters (K-Means)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# el algoritmo agrupo a los clientes en 5 grupos cuyas caracteristicas son:
#   1. Ganan bastante y gastan poco
#   2. Ganan bien y gastan ni mucho ni poco
#   3. Ganan bastante y gastan bastante (son a los que se deberia prestar mas atencion, los target)
#   4. Ganan poco y gastan mucho
#   5. Ganan poco y gastan poco