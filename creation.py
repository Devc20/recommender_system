import os
import zipfile
from utils.utils import download_file


def download_metadata():
    """Descarga y extrae el archivo de metadatos `tracks.csv`."""
    metadata_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"  # URL para `fma_metadata.zip`
    metadata_zip_path = os.path.join("data", "fma_metadata.zip")
    tracks_path = os.path.join("data", "tracks.csv")

    # Verifica si `tracks.csv` ya existe
    if os.path.exists(tracks_path):
        print(f"`tracks.csv` ya existe en la carpeta `data/`.")
        return

    # Descargar el archivo ZIP de metadatos
    print("Descargando metadatos de las canciones...")
    download_file(metadata_url, metadata_zip_path)

    # Extraer `tracks.csv` desde el archivo ZIP
    print("Extrayendo `tracks.csv` de `fma_metadata.zip`...")
    with zipfile.ZipFile(metadata_zip_path, 'r') as zip_ref:
        zip_ref.extract('fma_metadata/tracks.csv', 'data/')  # Cambia a la ruta correcta

    print(f"Metadatos guardados en {tracks_path}")
