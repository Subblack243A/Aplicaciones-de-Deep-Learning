import argparse
import os
import sys
import ffmpeg  # Requiere instalar 'ffmpeg-python' (pip install ffmpeg-python)

def convert_mp4_to_mp3(input_path: str, output_path: str = None) -> None:
    """
    Convierte un archivo MP4 a MP3 usando FFMPEG.
    
    Args:
        input_path: Ruta del archivo de video MP4.
        output_path: Ruta opcional para el archivo MP3 de salida.
    """
    # Validar que el archivo de entrada exista
    if not os.path.isfile(input_path):
        print(f"Error: El archivo '{input_path}' no existe.", file=sys.stderr)
        return

    # Si no se especifica salida, usar el mismo nombre pero con extensión .mp3
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mp3"

    try:
        # Configuración de FFMPEG:
        # - input: archivo de entrada
        # - output: archivo de salida
        #   - acodec='libmp3lame': Codec para MP3
        #   - audio_bitrate='192k': Calidad de audio (192 kbps)
        #   - vn=None: 'No Video' (ignorar el stream de video)
        # - overwrite_output(): Sobrescribir si ya existe
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='libmp3lame', audio_bitrate='192k', vn=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Exito: '{input_path}' -> '{output_path}'")
        
    except ffmpeg.Error as e:
        # Captura errores específicos de FFMPEG (ej. archivo corrupto, codec faltante)
        print(f"Error al convertir '{input_path}':", file=sys.stderr)
        print(e.stderr.decode(), file=sys.stderr)
    except Exception as e:
        # Captura cualquier otro error inexperado
        print(f"Error inesperado: {e}", file=sys.stderr)

def main():
    # Verificar argumentos de línea de comandos
    if len(sys.argv) < 2:
        print("Uso: python audio_converter.py <archivo_video.mp4>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    convert_mp4_to_mp3(input_path)

if __name__ == "__main__":
    main()