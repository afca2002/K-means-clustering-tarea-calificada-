import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class HousePriceKMeans:
    """
    Clase para aplicar KMeans a los datos de house_price.csv usando los atributos price, built_in y area.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.kmeans = None
        self.labels = None

    def load_data(self):
        """Carga los datos del archivo CSV y selecciona las columnas relevantes."""
        df = pd.read_csv(self.csv_path)
        self.data = df[['price', 'built_in', 'area']].values
        self.df = df

    def fit(self, n_clusters=2):
        """Ajusta el modelo KMeans con el número de clusters especificado."""
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.labels = self.kmeans.fit_predict(self.data)
        self.df['cluster'] = self.labels

    def show_results(self):
        """Muestra la clasificación de cada casa."""
        print(self.df[['price', 'built_in', 'area', 'cluster']])

    def plot_clusters(self):
        """Grafica los clusters usando price y area (para visualización 2D)."""
        plt.figure(figsize=(10,6))
        plt.scatter(self.df['price'], self.df['area'], c=self.df['cluster'], cmap='viridis')
        plt.xlabel('Price')
        plt.ylabel('Area')
        plt.title('KMeans Clustering (2 clusters)')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    kmeans_model = HousePriceKMeans('house_price.csv')
    kmeans_model.load_data()
    kmeans_model.fit(n_clusters=2)
    kmeans_model.show_results()
    kmeans_model.plot_clusters()
