# features/audio_features.py
import os
import numpy as np
import pandas as pd
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TCON, TRCK, TDRC
from config import AUDIO_PATH, MFCC_FEATURES_FILE, MFCC_COEFFICIENTS
from tqdm import tqdm
import librosa
from audioread import NoBackendError
from utils.utils import get_audio_path


class AudioFeatureExtractor:
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.metadata_list = []

    def extract_metadata(self, track_id):
        """Extrae metadatos como título, artista, álbum y género de un archivo MP3 basado en `track_id`."""
        audio_path = get_audio_path(self.audio_dir, track_id)
        try:
            audio = MP3(audio_path, ID3=ID3)
            metadata = {
                'track_id': track_id,
                'album': audio.tags.get('TALB', TIT2(encoding=3, text='')).text[0],
                'title': audio.tags.get('TIT2', TALB(encoding=3, text='')).text[0],
                'artist': audio.tags.get('TPE1', TPE1(encoding=3, text='')).text[0],
                'tracknumber': audio.tags.get('TRCK', TRCK(encoding=3, text='')).text[0],
                'genre': audio.tags.get('TCON', TCON(encoding=3, text='')).text[0],
                'date': audio.tags.get('TDRC', TDRC(encoding=3, text='')).text[0] if 'TDRC' in audio.tags else ''
            }
            return metadata
        except (FileNotFoundError, MutagenError) as e:
            print(f"Error al extraer metadatos de {audio_path}: {e}")
            return None

    def extract_features(self, track_id):
        """Extrae características de audio como MFCC, chroma y spectral contrast basado en `track_id`."""
        audio_path = get_audio_path(self.audio_dir, track_id)
        try:
            y, sr = librosa.load(audio_path, sr=None)
            features = {}
            features['mfcc'] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COEFFICIENTS), axis=1)
            features['chroma'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            return np.concatenate([features['mfcc'], features['chroma'], features['spectral_contrast']])
        except (FileNotFoundError, NoBackendError, Exception) as e:
            print(f"Error al procesar características de {audio_path}: {e}")
            return None

    def process_new_audio_files(self, existing_ids):
        """Procesa solo archivos de audio nuevos y devuelve metadatos y características."""
        new_features = []
        new_metadata = []

        for subdir, _, files in os.walk(self.audio_dir):
            for file_name in tqdm(files, desc=f"Procesando archivos nuevos en {subdir}"):
                if file_name.endswith('.mp3'):
                    track_id = int(file_name.split('.')[0])

                    if track_id not in existing_ids:  # Procesar solo archivos nuevos
                        metadata = self.extract_metadata(track_id)
                        if metadata is not None:
                            new_metadata.append(metadata)

                        features = self.extract_features(track_id)
                        if features is not None:
                            new_features.append(features)

        return np.array(new_features), pd.DataFrame(new_metadata)


def get_or_create_mfcc_features(audio_dir):
    """Carga o crea el archivo mfcc_features.npy con las características y metadatos."""
    if os.path.exists(MFCC_FEATURES_FILE):
        data = np.load(MFCC_FEATURES_FILE, allow_pickle=True).item()
        print("Cargando características desde mfcc_features.npy")
        features = data['features']
        metadata = pd.DataFrame(data['metadata'])
    else:
        print("Extrayendo características iniciales de audio y guardando en mfcc_features.npy")
        extractor = AudioFeatureExtractor(audio_dir)
        features, metadata = extractor.process_new_audio_files(set())
        np.save(MFCC_FEATURES_FILE, {'features': features, 'metadata': metadata.to_dict(orient='records')})

    return features, metadata


def update_mfcc_features(audio_dir):
    """Actualiza mfcc_features.npy con características de nuevas canciones."""
    # Cargar el archivo existente si existe
    if os.path.exists(MFCC_FEATURES_FILE):
        data = np.load(MFCC_FEATURES_FILE, allow_pickle=True).item()
        existing_features = data['features']
        existing_metadata = pd.DataFrame(data['metadata'])
        existing_ids = set(existing_metadata['track_id'])
    else:
        existing_features = np.array([])
        existing_metadata = pd.DataFrame()
        existing_ids = set()

    # Extraer características para archivos nuevos
    extractor = AudioFeatureExtractor(audio_dir)
    new_features, new_metadata = extractor.process_new_audio_files(existing_ids)

    # Si hay nuevas características, combinar y guardar
    if len(new_features) > 0:
        all_features = np.vstack([existing_features, new_features]) if existing_features.size else new_features
        all_metadata = pd.concat([existing_metadata, new_metadata], ignore_index=True)

        # Guardar el archivo actualizado
        np.save(MFCC_FEATURES_FILE, {'features': all_features, 'metadata': all_metadata.to_dict(orient='records')})
        print("Características de audio actualizadas en mfcc_features.npy")
    else:
        print("No se encontraron canciones nuevas.")

    return all_features, all_metadata
