"""
AudioProcessor: Carga, normalización y transformación de audio a espectrogramas.
Utiliza librosa para el procesamiento de audio.
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioProcessor:
    """Procesa archivos de audio: carga, normaliza, genera espectrogramas."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 160,
        fmin: float = 0.0,
        fmax: float = 8000.0,
    ):
        """
        Inicializa el AudioProcessor con los parámetros de procesamiento.

        Args:
            sample_rate: Tasa de muestreo.
            n_mels: Número de bandas mel.
            n_fft: Tamaño de la FFT.
            hop_length: Longitud del salto entre ventanas.
            fmin: Frecuencia mínima.
            fmax: Frecuencia máxima.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """
        Carga un archivo de audio y lo convierte a un array de NumPy.

        Args:
            audio_path: Ruta del archivo de audio.

        Returns:
            Tupla (audio_array, sample_rate).

        Raises:
            FileNotFoundError: Si el archivo no existe.
            RuntimeError: Si ocurre un error al cargar.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"El archivo '{audio_path}' no existe.")

        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error al cargar el audio '{audio_path}': {e}")

    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """
        Normaliza el audio con media 0 y desviación estándar 1.

        Args:
            audio: Array de NumPy con el audio.

        Returns:
            Array de NumPy con el audio normalizado.
        """
        mean = np.mean(audio)
        std = np.std(audio)
        return (audio - mean) / (std + 1e-8)

    def to_mel_spectrogram(self, audio: np.ndarray, power: float = 2.0) -> np.ndarray:
        """
        Convierte un array de audio a espectrograma mel en decibelios.

        Args:
            audio: Array de NumPy con el audio.
            power: Potencia del espectrograma.

        Returns:
            Array de NumPy con el espectrograma mel en dB.
        """
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=power,
        )
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_db

    def prepare_for_model(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Prepara el espectrograma para el modelo:
        normaliza, transpone (tiempo x frecuencia) y añade dimensión de batch.

        Args:
            mel_db: Espectrograma mel en dB (n_mels x time_steps).

        Returns:
            Array con shape (1, time_steps, n_mels) listo para el modelo.
        """
        mel_db = self.normalize(mel_db)
        # Transponer: (n_mels, time) -> (time, n_mels)
        mel_db = mel_db.T
        # Añadir dimensión de batch: (1, time, n_mels)
        mel_db = np.expand_dims(mel_db, axis=0)
        return mel_db

    def process_audio_file(self, audio_path: str) -> np.ndarray:
        """
        Pipeline completo: carga audio → espectrograma → preparado para modelo.

        Args:
            audio_path: Ruta del archivo de audio.

        Returns:
            Array preparado para el modelo con shape (1, time_steps, n_mels).
        """
        audio, sr = self.load_audio(audio_path)
        mel_db = self.to_mel_spectrogram(audio)
        return self.prepare_for_model(mel_db)

    def save_spectrogram(self, mel_db: np.ndarray, output_path: str) -> str:
        """
        Guarda una visualización del espectrograma como imagen.

        Args:
            mel_db: Espectrograma mel en dB.
            output_path: Ruta del archivo de imagen de salida.

        Returns:
            Ruta del archivo guardado.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            mel_db, sr=self.sample_rate, x_axis='time', y_axis='mel',
            hop_length=self.hop_length
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Espectrograma guardado en '{output_path}'")
        return output_path
