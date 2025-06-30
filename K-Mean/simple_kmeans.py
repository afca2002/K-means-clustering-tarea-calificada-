import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class SimpleKMeansExample:
    """
    Clase para aplicar el método del codo y KMeans a un conjunto de datos bidimensionales.
    """
    def __init__(self):
        self.data = np.array([
            [1, 1],
            [2, 4],
            [3, 2],
            [3, 5],
            [4, 4],
            [4, 7],
            [6, 4],
            [6, 6]
        ])
        self.kmeans = None
        self.labels = None

    def elbow_method(self, max_k=7):
        """Grafica el método del codo para elegir el k óptimo."""
        sse = []
        k_rango = range(1, max_k+1)
        for k in k_rango:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(self.data)
            sse.append(kmeans.inertia_)
        plt.figure(figsize=(10,6))
        plt.plot(k_rango, sse, '-o')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('SSE')
        plt.title('Método del codo para determinar k óptimo')
        plt.grid(True)
        plt.show()

    def fit(self, n_clusters):
        """Ajusta el modelo KMeans con el número de clusters especificado."""
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.labels = self.kmeans.fit_predict(self.data)

    def show_classification(self):
        """Muestra la clasificación de cada caso."""
        for i, label in enumerate(self.labels, 1):
            print(f'Caso {i}: Cluster {label}')

if __name__ == "__main__":
    example = SimpleKMeansExample()
    example.elbow_method()
    # Suponiendo k óptimo = 2 (ajustar según el gráfico)
    example.fit(n_clusters=2)
    example.show_classification()
