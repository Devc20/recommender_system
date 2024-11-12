# config.py

# Ruta principal donde se almacenan los archivos de audio en subcarpetas
AUDIO_PATH = 'data/fma_small/'

# Archivo .npy para almacenar características de audio extraídas
MFCC_FEATURES_FILE = 'data/mfcc_features.npy'

# Archivos de índice para KD-Tree y HNSW
KD_TREE_INDEX_FILE = 'data/kd_tree_index.pkl'    # Archivo de índice para KD-Tree
HNSW_INDEX_FILE = 'data/hnsw_index.bin'          # Archivo de índice para HNSW

# Parámetros para la extracción de características de audio
MFCC_COEFFICIENTS = 20                           # Número de coeficientes MFCC a extraer
