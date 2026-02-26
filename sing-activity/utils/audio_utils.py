import librosa
import numpy as np
import torch

# ============================================================
# MÓDULO 2: Extracción de Espectrogramas Mel
# ============================================================

class AudioProcessor:
    """
    Procesa archivos de audio para extraer espectrogramas Mel.
    El espectrograma Mel es la representación intermedia que la red
    neuronal aprende a generar a partir del texto.
    """
    def __init__(
        self,
        sample_rate: int = 22050,     # Frecuencia de muestreo estándar para TTS/SVS
        n_fft: int = 1024,            # Tamaño de la FFT
        hop_length: int = 256,        # Salto entre ventanas (determina resolución temporal)
        n_mels: int = 80,             # Número de bandas Mel (dimensión del espectrograma)
        fmin: float = 0.0,            # Frecuencia mínima del filtro Mel
        fmax: float = 8000.0          # Frecuencia máxima del filtro Mel
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def load_audio(self, path: str) -> np.ndarray:
        """Carga audio y lo resamplea a la frecuencia objetivo."""
        wav, sr = librosa.load(path, sr=self.sample_rate)
        wav = wav / (max(abs(wav)) + 1e-7)  # Normalización
        return wav

    def get_mel_spectrogram(self, wav: np.ndarray) -> np.ndarray:
        """
        Extrae espectrograma Mel del audio.
        Forma de salida: (n_mels, T) donde T = frames temporales.
        """
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        # Conversión a escala logarítmica (dB) para mejor rango dinámico
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def mel_to_audio(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Reconstrucción aproximada usando Griffin-Lim.
        Usado como vocoder básico de referencia.
        """
        mel = librosa.db_to_power(mel_db)
        wav = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            n_iter=60  # Más iteraciones = mejor calidad
        )
        return wav

def extract_pitch(wav_path, sr=22050, hop_length=256):
    """
    Extrae F0 usando pyin (librosa).
    Crítico para canto: el pitch define la melodía.
    """
    wav, _ = librosa.load(wav_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        wav, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )
    # Reemplazar NaN (zonas no vocalizadas) con 0
    f0 = np.nan_to_num(f0)
    return f0

def extract_energy(wav_path, sr=22050, hop_length=256, n_fft=1024):
    """Extrae energía RMS por frame."""
    wav, _ = librosa.load(wav_path, sr=sr)
    energy = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=hop_length)[0]
    return energy
