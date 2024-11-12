# indexers/hnsw_index.py
import hnswlib
import os
from config import HNSW_INDEX_FILE


class HNSWIndex:
    def __init__(self, features, ef=200, M=16, space='cosine'):
        self.dim = features.shape[1]
        self.index = hnswlib.Index(space, self.dim)

        # Inicializa el índice o carga el índice existente
        if os.path.exists(HNSW_INDEX_FILE):
            self.index.load_index(HNSW_INDEX_FILE)
            print("HNSW index cargado desde archivo.")
            self.num_elements = self.index.element_count
        else:
            self.index.init_index(max_elements=features.shape[0] + 10000, ef_construction=ef, M=M)
            self.index.add_items(features)
            self.index.save_index(HNSW_INDEX_FILE)
            print("HNSW index creado y guardado en archivo.")
            self.num_elements = features.shape[0]

    def update_index(self, new_features):
        """Actualiza el índice HNSW con nuevas características."""
        # Incrementa `num_elements` según la cantidad de nuevas características
        self.index.add_items(new_features)
        self.num_elements += new_features.shape[0]
        # Guarda el índice actualizado
        self.index.save_index(HNSW_INDEX_FILE)
        print("HNSW index actualizado con nuevas características.")

    def search_with_distances(self, query_vector, k=5):
        """Realiza una búsqueda en el índice HNSW para encontrar los `k` vecinos más cercanos con distancias."""
        labels, distances = self.index.knn_query(query_vector, k=k)
        return distances[0], labels[0]
