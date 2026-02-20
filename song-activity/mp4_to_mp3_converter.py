import argparse
import os
import sys
import subprocess

def convert_mp4_to_mp3(input_path: str, output_path: str = None) -> None:
    """
    Convierte un archivo MP4 a MP3 usando FFMPEG directamente a través de subprocess.
    
    Args:
        input_path: Ruta del archivo de video MP4.
        output_path: Ruta opcional para el archivo MP3 de salida.
    """
    if not os.path.isfile(input_path):
        print(f"Error: El archivo '{input_path}' no existe.", file=sys.stderr)
        return

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mp3"
    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-ab', '96k',
        '-ar', '16000',
        '-ac', '1',
        output_path
    ]

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Exito: '{input_path}' -> '{output_path}'")
        
    except subprocess.CalledProcessError as e:
        print(f"Error al convertir '{input_path}':", file=sys.stderr)
        print(e.stderr.decode(), file=sys.stderr)
    except FileNotFoundError:
        print("Error: No se encontró el ejecutable 'ffmpeg'.", file=sys.stderr)
    except Exception as e:
        print(f"Error inesperado: {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("Uso: python mp4_converter.py <archivo_video.mp4>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    convert_mp4_to_mp3(input_path)

if __name__ == "__main__":
    main()