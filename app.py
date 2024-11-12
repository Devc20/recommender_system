from flask import Flask, request, render_template
from features.audio_features import get_or_create_mfcc_features, update_mfcc_features
from indexers.kd_tree_index import KDTreeIndex
from indexers.hnsw_index import HNSWIndex
from config import AUDIO_PATH
import numpy as np

app = Flask(__name__)

# Cargar características y metadatos al iniciar
features, metadata = get_or_create_mfcc_features(AUDIO_PATH)
kd_tree_index = KDTreeIndex(features)
hnsw_index = HNSWIndex(features)

def recommend_songs(song_index, query_vector, k=5):
    """Genera recomendaciones de canciones similares utilizando KD-Tree y HNSW."""
    recommendations = {}

    # Obtener recomendaciones con KD-Tree
    kd_tree_distances, kd_tree_indices = kd_tree_index.search_with_distances(query_vector, k + 1)
    max_distance = max(kd_tree_distances) if len(kd_tree_distances) > 0 else 1
    recommendations['kd_tree'] = [
        {
            **metadata.iloc[idx].to_dict(),
            'score': f"{(1 - (dist / max_distance)) * 100:.2f}%"
        }
        for dist, idx in zip(kd_tree_distances, kd_tree_indices)
        if idx != song_index  # Excluir la canción seleccionada
    ][:k]  # Limitar a las primeras k recomendaciones

    # Obtener recomendaciones con HNSW
    hnsw_distances, hnsw_indices = hnsw_index.search_with_distances(query_vector, k + 1)
    max_distance_hnsw = max(hnsw_distances) if len(hnsw_distances) > 0 else 1
    recommendations['hnsw'] = [
        {
            **metadata.iloc[idx].to_dict(),
            'score': f"{(1 - (dist / max_distance_hnsw)) * 100:.2f}%"
        }
        for dist, idx in zip(hnsw_distances, hnsw_indices)
        if idx != song_index  # Excluir la canción seleccionada
    ][:k]

    return recommendations

def update_system_with_new_songs(audio_dir):
    """Actualiza las características y los índices de similitud si se agregan nuevas canciones."""
    global features, metadata
    # Actualizar mfcc_features.npy con nuevas canciones
    features, metadata = update_mfcc_features(audio_dir)

    # Actualizar los índices de similitud con las nuevas características
    kd_tree_index.update_index(features)
    hnsw_index.update_index(features)
    print("Sistema actualizado con nuevas canciones.")

@app.route('/')
def index():
    """Página principal con la lista de canciones disponibles para recomendar."""
    return render_template('index.html', songs=metadata.to_dict(orient='records'), recommendations=None)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint para generar recomendaciones basadas en la canción seleccionada por el usuario."""
    selected_song = int(request.form['song_index'])
    query_vector = features[selected_song]
    recommendations = recommend_songs(selected_song, query_vector, k=5)

    return render_template(
        'index.html',
        songs=metadata.to_dict(orient='records'),
        recommendations=recommendations,
        selected_song=selected_song
    )

@app.route('/update', methods=['POST'])
def update():
    """Actualiza el sistema cuando se agregan nuevas canciones."""
    update_system_with_new_songs(AUDIO_PATH)
    return "Sistema actualizado con nuevas canciones.", 200

if __name__ == '__main__':
    app.run(debug=True)
