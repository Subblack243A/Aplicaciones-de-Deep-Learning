import librosa
import soundfile as sf
import numpy as np
import os
import math
from scipy.signal import butter, sosfilt

# =============================================================
# Parámetros de audio requeridos por el proyecto (AudioProcessor)
# =============================================================
TARGET_SR = 22050       # Frecuencia de muestreo requerida por Tacotron 2
F_MIN = 80.0            # Frecuencia mínima del filtro Mel (Hz)
F_MAX = 8000.0          # Frecuencia máxima del filtro Mel (Hz)
TARGET_DB = -3.0        # dB máximo de pico (headroom para el vocoder)


def normalize_peak(y: np.ndarray, target_db: float = TARGET_DB) -> np.ndarray:
    """Normaliza el audio para que el pico no supere target_db."""
    peak = np.max(np.abs(y)) + 1e-9
    target_amplitude = 10 ** (target_db / 20.0)
    return y * (target_amplitude / peak)


def highpass_filter(y: np.ndarray, sr: int, cutoff_hz: float = F_MIN) -> np.ndarray:
    """
    Aplica un filtro pasa-altos Butterworth de orden 4.
    Elimina frecuencias por debajo de cutoff_hz (ruido de fondo, rumble).
    """
    sos = butter(4, cutoff_hz / (sr / 2), btype='high', output='sos')
    return sosfilt(sos, y).astype(np.float32)


def preprocess_audio(input_path: str) -> tuple[np.ndarray, int]:
    """
    Preprocessing completo del audio antes de usarlo en entrenamiento:
        1. Carga y resamplea a TARGET_SR (22050 Hz)
        2. Convierte a mono si es estéreo
        3. Aplica filtro pasa-altos (elimina ruido < F_MIN)
        4. Normaliza el pico a TARGET_DB
    """
    print(f"Cargando: {input_path}")
    y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
    print(f"  → Cargado a {TARGET_SR} Hz, mono, {len(y)} muestras ({len(y)/TARGET_SR:.2f} segundos)")

    print(f"  → Aplicando filtro pasa-altos ({F_MIN} Hz)...")
    y = highpass_filter(y, TARGET_SR, F_MIN)

    print(f"  → Normalizando pico a {TARGET_DB} dB...")
    y = normalize_peak(y, TARGET_DB)

    return y, TARGET_SR


def split_wav(input_path: str, num_segments: int = 5):
    """
    Pre-procesa y divide un archivo WAV en N partes iguales.
    Los segmentos se guardan en la misma carpeta que el archivo original.
    """
    if not os.path.exists(input_path):
        print(f"Error: No se encontró el archivo {input_path}")
        return

    y, sr = preprocess_audio(input_path)

    total_samples = len(y)
    samples_per_segment = math.ceil(total_samples / num_segments)

    base_dir  = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    print(f"\nDividiendo en {num_segments} segmentos de ~{samples_per_segment/sr:.2f} s...")
    for i in range(num_segments):
        start   = i * samples_per_segment
        end     = min((i + 1) * samples_per_segment, total_samples)
        segment = y[start:end]

        output_name = f"{base_name}_{i+1:02d}.wav"
        output_path = os.path.join(base_dir, output_name)
        sf.write(output_path, segment, sr, subtype='PCM_16')
        print(f"  Guardado: {output_path}  ({len(segment)/sr:.2f} s)")

    print("\n¡Procesamiento completado!")


if __name__ == "__main__":
    audio_path = "/home/subblack/uni/ADL/Aplicaciones-de-Deep-Learning/sing-activity/dataset/wavs/snuff.wav"
    split_wav(audio_path, num_segments=5)
