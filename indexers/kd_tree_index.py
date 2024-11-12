# indexers/kd_tree_index.py
from sklearn.neighbors import KDTree
import pickle
import os
from config import KD_TREE_INDEX_FILE

class KDTreeIndex:
    def __init__(self, features):
        self.features = features
        self.build_index()

    def build_index(self):
        """Construye o reconstruye el índice KD-Tree y lo guarda en un archivo."""
        self.index = KDTree(self.features)
        with open(KD_TREE_INDEX_FILE, 'wb') as f:
            pickle.dump(self.index, f)
        print("KD-Tree index creado y guardado en archivo.")

    def update_index(self, new_features):
        """Actualiza el índice KD-Tree con nuevas características agregando nuevas canciones."""
        # Agregar las nuevas características a las existentes
        self.features = np.vstack([self.features, new_features])
        # Recrear el índice KD-Tree con las características actualizadas
        self.build_index()

    def search_with_distances(self, query_vector, k=5):
        """Realiza una búsqueda en el índice KD-Tree para encontrar los `k` vecinos más cercanos con distancias."""
        distances, indices = self.index.query([query_vector], k=k)
        return distances[0], indices[0]
