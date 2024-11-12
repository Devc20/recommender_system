# utils/utils.py
import os
import requests


def get_audio_path(base_dir, track_id):
    """
    Genera la ruta completa al archivo de audio usando el ID de la pista.

    Los archivos de audio están organizados en subcarpetas en función de los primeros tres dígitos del ID.
    Por ejemplo, para track_id=123456, la ruta sería "base_dir/123/123456.mp3".
    """
    track_id_str = f"{track_id:06d}"  # Formatea el ID a 6 dígitos con ceros a la izquierda
    folder = track_id_str[:3]  # Usa los primeros tres dígitos para definir la subcarpeta
    return os.path.join(base_dir, folder, track_id_str + ".mp3")


def download_file(url, dest_path):
    """
    Descarga un archivo desde una URL y lo guarda en `dest_path`.

    La función descarga el contenido en bloques de 1024 bytes y maneja posibles errores de conexión.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Verifica si la solicitud fue exitosa
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Crea el directorio si no existe
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Archivo descargado: {dest_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar {url}: {e}")
