"""
Predictor: Carga un modelo ASR entrenado y transcribe archivos de audio/video.
Soporta decodificación greedy y beam search.
"""

import os
import numpy as np
import tensorflow as tf

from text_encoder import TextEncoder
from audio_processor import AudioProcessor
from audio_converter import AudioConverter


class Predictor:
    """
    Carga un modelo ASR entrenado y genera transcripciones
    a partir de archivos de audio o video.
    """

    def __init__(
        self,
        model_path: str,
        encoder: TextEncoder = None,
        processor: AudioProcessor = None,
    ):
        """
        Args:
            model_path: Ruta al modelo guardado (.keras o directorio SavedModel).
            encoder: Instancia de TextEncoder (default: se crea uno nuevo).
            processor: Instancia de AudioProcessor (default: se crea uno nuevo).
        """
        self.encoder = encoder or TextEncoder()
        self.processor = processor or AudioProcessor()
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> tf.keras.Model:
        """Carga el modelo desde disco."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en '{model_path}'")
        model = tf.keras.models.load_model(model_path)
        print(f"Modelo cargado desde '{model_path}'")
        return model

    def transcribe_audio(self, audio_path: str, beam_width: int = None) -> str:
        """
        Transcribe un archivo de audio.

        Args:
            audio_path: Ruta al archivo de audio (WAV, FLAC, etc.).
            beam_width: Si se proporciona, usa beam search con este ancho.
                       Si es None, usa decodificación greedy.

        Returns:
            Texto transcrito.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Archivo de audio no encontrado: '{audio_path}'")

        # Procesar audio → espectrograma → preparar para modelo
        prepared = self.processor.process_audio_file(audio_path)

        # Predicción
        y_pred = self.model.predict(prepared, verbose=0)

        # Decodificar
        if beam_width:
            return self.encoder.decode_beam(y_pred, beam_width)
        response = self.encoder.decode_greedy(y_pred, audio_path)
        return response

    def transcribe_video(self, video_path: str, beam_width: int = None) -> str:
        """
        Transcribe un archivo de video (convierte a WAV primero).

        Args:
            video_path: Ruta al archivo de video.
            beam_width: Ancho de beam search (None = greedy).

        Returns:
            Texto transcrito.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Archivo de video no encontrado: '{video_path}'")

        # Convertir video a WAV temporal
        wav_path = AudioConverter.any_to_wav(video_path)
        try:
            result = self.transcribe_audio(wav_path, beam_width)
            return result
        finally:
            # Limpiar WAV temporal (solo si fue generado, no el original)
            ext = os.path.splitext(video_path)[1].lower()
            if ext != '.wav' and os.path.isfile(wav_path):
                os.remove(wav_path)

    def transcribe(self, input_path: str, beam_width: int = None) -> str:
        """
        Transcribe un archivo de audio o video (detecta automáticamente el tipo).

        Args:
            input_path: Ruta al archivo.
            beam_width: Ancho de beam search (None = greedy).

        Returns:
            Texto transcrito.
        """
        ext = os.path.splitext(input_path)[1].lower()
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

        if ext in video_exts:
            return self.transcribe_video(input_path, beam_width)
        else:
            # Convertir a WAV si no lo es
            wav_path = AudioConverter.any_to_wav(input_path)
            try:
                return self.transcribe_audio(wav_path, beam_width)
            finally:
                if ext != '.wav' and os.path.isfile(wav_path) and wav_path != input_path:
                    os.remove(wav_path)



    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calcula la tasa de error de palabras (Word Error Rate).

        Args:
            reference: Texto de referencia.
            hypothesis: Texto transcrito (hipótesis).

        Returns:
            WER como float (0.0 = perfecto, 1.0 = 100% error).
        """
        try:
            from jiwer import wer
            return wer(reference, hypothesis)
        except ImportError:
            print("Instala 'jiwer' para calcular WER: pip install jiwer")
            return None
