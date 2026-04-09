import numpy as np
import librosa
import soundfile as sf
import scipy.ndimage
import matplotlib.pyplot as plt

def apply_bandpass_filter(D, sr, f_min=140, f_max=None):
    """
    Aplica un filtro pasa-banda manual en el dominio de la frecuencia (STFT).
    Elimina ruido de baja frecuencia (rumble < 140Hz) que enturbia la voz.
    
    Args:
        D: Matriz STFT (compleja).
        sr: Tasa de muestreo.
        f_min: Frecuencia de corte inferior (Hz). Default 140Hz.
        f_max: Frecuencia de corte superior (Hz). Default Nyquist.
    """
    n_fft = 2 * (D.shape[0] - 1)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    if f_max is None:
        f_max = sr // 2
        
    mask = (frequencies >= f_min) & (frequencies <= f_max)
    
    return D * mask[:, np.newaxis]

def hpss_manual(y, sr, margin_harmonic=1.0, margin_percussive=1.0, kernel_size=31):
    """
    Implementación manual de la Separación de Fuentes Armónicas y Percusivas (HPSS)
    usando filtros de mediana y máscaras suaves (Wiener Filter).
    Incluye pre-filtrado de banda para "limpiar" la voz.
    """
    D = librosa.stft(y)
    D = apply_bandpass_filter(D, sr, f_min=140)
    S = np.abs(D)
    phase = np.exp(1.j * np.angle(D))
    
    H_filter = scipy.ndimage.median_filter(S, size=(1, kernel_size))
    P_filter = scipy.ndimage.median_filter(S, size=(kernel_size, 1))
    
    eps = np.finfo(float).eps
    mask_h = (H_filter * margin_harmonic) / (H_filter * margin_harmonic + P_filter * margin_percussive + eps)
    mask_p = (P_filter * margin_percussive) / (H_filter * margin_harmonic + P_filter * margin_percussive + eps)
    
    D_harmonic = D * mask_h
    D_percussive = D * mask_p
    
    y_harmonic = librosa.istft(D_harmonic)
    y_percussive = librosa.istft(D_percussive)
    
    min_len = min(len(y), len(y_harmonic))
    y_music_no_voice = y[:min_len] - y_harmonic[:min_len]
    
    return y_harmonic, y_percussive, y_music_no_voice

def hz_to_mel(freq):
    """Fórmula manual para convertir Hz a Mel."""
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_hz(mel):
    """Fórmula manual para convertir Mel a Hz."""
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def mel_filterbank_manual(n_fft, n_mels, sr, fmin, fmax):
    """
    Construcción manual de la matriz de filtros Mel triangular.
    
    Args:
        n_fft: Tamaño de la FFT.
        n_mels: Número de bandas Mel.
        sr: Tasa de muestreo.
        fmin: Frecuencia mínima.
        fmax: Frecuencia máxima.
    
    Returns:
        Matriz de filtros (n_mels, 1 + n_fft // 2)
    """
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    n_freqs = 1 + n_fft // 2
    weights = np.zeros((n_mels, n_freqs))
    
    for i in range(n_mels):
        b_left = bins[i]
        b_center = bins[i+1]
        b_right = bins[i+2]
        
        for k in range(b_left, b_center):
            weights[i, k] = (k - b_left) / (b_center - b_left)
            
        for k in range(b_center, b_right):
            weights[i, k] = (b_right - k) / (b_right - b_center)
            
    return weights

def mel_spectrogram_manual(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Genera un espectrograma de Mel manualmente.
    """
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    mel_basis = mel_filterbank_manual(n_fft, n_mels, sr, 0, sr/2)
    mel_S = np.dot(mel_basis, D)
    mel_S_db = 10.0 * np.log10(mel_S + 1e-10)
    return mel_S_db
