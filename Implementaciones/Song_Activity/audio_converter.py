"""
AudioConverter: Conversión de formatos de audio/video (MP4→MP3→WAV).
Utiliza ffmpeg-python como backend.
"""

import os
import sys
import ffmpeg


class AudioConverter:
    """Convierte archivos de audio/video entre formatos usando FFMPEG."""

    @staticmethod
    def mp4_to_mp3(input_path: str, output_path: str = None) -> str:
        """
        Convierte un archivo MP4 a MP3.

        Args:
            input_path: Ruta del archivo MP4.
            output_path: Ruta opcional para el MP3 de salida.

        Returns:
            Ruta del archivo MP3 generado.

        Raises:
            FileNotFoundError: Si el archivo de entrada no existe.
            RuntimeError: Si ocurre un error durante la conversión.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"El archivo '{input_path}' no existe.")

        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".mp3"

        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='libmp3lame', audio_bitrate='192k', vn=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"Conversión exitosa: '{input_path}' -> '{output_path}'")
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error FFMPEG al convertir '{input_path}': {e.stderr.decode()}")

    @staticmethod
    def mp3_to_wav(input_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
        """
        Convierte un archivo MP3 a WAV mono.

        Args:
            input_path: Ruta del archivo MP3.
            output_path: Ruta opcional para el WAV de salida.
            sample_rate: Tasa de muestreo deseada (default: 16000 Hz).

        Returns:
            Ruta del archivo WAV generado.

        Raises:
            FileNotFoundError: Si el archivo de entrada no existe.
            RuntimeError: Si ocurre un error durante la conversión.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"El archivo '{input_path}' no existe.")

        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".wav"

        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='pcm_s16le', ar=str(sample_rate), ac=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"Conversión exitosa: '{input_path}' -> '{output_path}'")
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error FFMPEG al convertir '{input_path}': {e.stderr.decode()}")

    @staticmethod
    def mp4_to_wav(input_path: str, sample_rate: int = 16000) -> str:
        """
        Convierte un archivo MP4 directamente a WAV (MP4→MP3→WAV).

        Args:
            input_path: Ruta del archivo MP4.
            sample_rate: Tasa de muestreo deseada.

        Returns:
            Ruta del archivo WAV generado.
        """
        mp3_path = AudioConverter.mp4_to_mp3(input_path)
        wav_path = AudioConverter.mp3_to_wav(mp3_path, sample_rate=sample_rate)
        # Limpiar el archivo MP3 intermedio
        if os.path.isfile(mp3_path):
            os.remove(mp3_path)
        return wav_path

    @staticmethod
    def any_to_wav(input_path: str, sample_rate: int = 16000) -> str:
        """
        Convierte cualquier archivo de audio/video soportado a WAV mono.

        Args:
            input_path: Ruta del archivo de entrada.
            sample_rate: Tasa de muestreo deseada.

        Returns:
            Ruta del archivo WAV generado, o la misma ruta si ya es WAV.
        """
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.wav':
            return input_path
        elif ext == '.mp4':
            return AudioConverter.mp4_to_wav(input_path, sample_rate)
        elif ext == '.mp3':
            return AudioConverter.mp3_to_wav(input_path, sample_rate=sample_rate)
        else:
            # Intentar conversión directa con ffmpeg
            output_path = os.path.splitext(input_path)[0] + ".wav"
            try:
                (
                    ffmpeg
                    .input(input_path)
                    .output(output_path, acodec='pcm_s16le', ar=str(sample_rate), ac=1)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                return output_path
            except ffmpeg.Error as e:
                raise RuntimeError(f"No se pudo convertir '{input_path}': {e.stderr.decode()}")
