import numpy as np
import librosa
import soundfile as sf
import scipy.ndimage
import matplotlib.pyplot as plt

def hpss_manual(y, sr, margin_harmonic=1.0, margin_percussive=1.0, kernel_size=31):
    """
    Implementación manual de la Separación de Fuentes Armónicas y Percusivas (HPSS)
    usando filtros de mediana.
    
    Args:
        y: Señal de audio.
        sr: Tasa de muestreo.
        margin_harmonic: Margen para la máscara armónica.
        margin_percussive: Margen para la máscara percusiva.
        kernel_size: Tamaño del filtro de mediana.
    """
    # 1. Calcular STFT (Short-Time Fourier Transform)
    # D es una matriz compleja: Tiempo x Frecuencia
    D = librosa.stft(y)
    
    # 2. Separar Magnitud (S) y Fase (P)
    # Usamos la magnitud para el filtrado, y la fase para reconstruir el audio después
    S = np.abs(D)
    phase = np.exp(1.j * np.angle(D))
    
    # 3. Aplicar Filtros de Mediana
    # Filtro Horizontal: Suaviza en el tiempo -> Resalta líneas horizontales (Armónicos)
    # size=(1, kernel_size) significa 1 pixel de alto x kernel_size pixels de ancho
    H_filter = scipy.ndimage.median_filter(S, size=(1, kernel_size))
    
    # Filtro Vertical: Suaviza en frecuencia -> Resalta líneas verticales (Percusivos)
    # size=(kernel_size, 1) significa kernel_size pixels de alto x 1 pixel de ancho
    P_filter = scipy.ndimage.median_filter(S, size=(kernel_size, 1))
    
    # 4. Crear Máscaras (Soft Masks)
    # Comparamos las imágenes filtradas para decidir qué píxel pertenece a qué fuente.
    # Epsilons para evitar división por cero si fuera necesario, aunque aquí comparamos directamente.
    
    # Máscara Armónica: H > P
    mask_h = (H_filter * margin_harmonic) > P_filter
    
    # Máscara Percusiva: P > H
    mask_p = (P_filter * margin_percussive) > H_filter
    
    # Nota: Esta es una máscara binaria (Hard Mask). Para Soft Mask usaríamos el ratio Wiener.
    # Para fines educativos, la máscara binaria es más clara de entender.
    
    # Aseguramos que son float para multiplicar
    mask_h = mask_h.astype(float)
    mask_p = mask_p.astype(float)
    
    # 5. Aplicar Máscaras a la STFT original
    D_harmonic = D * mask_h
    D_percussive = D * mask_p
    
    # 6. Reconstruir Audio (Inverse STFT)
    y_harmonic = librosa.istft(D_harmonic)
    y_percussive = librosa.istft(D_percussive)
    
    return y_harmonic, y_percussive

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
    # 1. Definir puntos en escala Mel
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    
    # Puntos equidistantes en escala Mel
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # 2. Convertir Hz a índices (bins) de FFT
    # bin = freq * (n_fft + 1) / sr  ... aproximación básica
    # bin = floor((n_fft + 1) * freq / sr)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # 3. Construir la matriz de filtros
    n_freqs = 1 + n_fft // 2
    weights = np.zeros((n_mels, n_freqs))
    
    for i in range(n_mels):
        # Indices del filtro triangular actual: start, center, end
        b_left = bins[i]
        b_center = bins[i+1]
        b_right = bins[i+2]
        
        # Pendiente ascendente (izquierda a centro)
        for k in range(b_left, b_center):
            weights[i, k] = (k - b_left) / (b_center - b_left)
            
        # Pendiente descendente (centro a derecha)
        for k in range(b_center, b_right):
            weights[i, k] = (b_right - k) / (b_right - b_center)
            
    return weights

def mel_spectrogram_manual(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Genera un espectrograma de Mel manualmente.
    """
    # 1. Calcular Espectrograma de Potencia (Power Spectrogram)
    # |STFT|^2
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    
    # 2. Crear Filtros Mel
    mel_basis = mel_filterbank_manual(n_fft, n_mels, sr, 0, sr/2)
    
    # 3. Aplicar Filtros (Producto Punto)
    # (n_mels, n_freqs) x (n_freqs, time_steps) -> (n_mels, time_steps)
    mel_S = np.dot(mel_basis, D)
    
    # 4. Escala Logarítmica (Decibeles)
    # Añadimos una pequeña constante (1e-6) para evitar log(0)
    mel_S_db = 10.0 * np.log10(mel_S + 1e-10)
    
    return mel_S_db

if __name__ == "__main__":
    # Ejemplo de uso simple si se ejecuta directamente
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Procesando: {input_file}")
        
        # Cargar audio
        y, sr = librosa.load(input_file)
        
        # 1. HPSS
        print("Ejecutando HPSS Manual...")
        y_harm, y_perc = hpss_manual(y, sr)
        sf.write('harmonic.wav', y_harm, sr)
        sf.write('percussive.wav', y_perc, sr)
        print("Guardados 'harmonic.wav' y 'percussive.wav'")
        
        # 2. Mel Spectrogram
        print("Generando Espectrograma Mel Manual...")
        mel_spec = mel_spectrogram_manual(y, sr)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
        plt.title('Espectrograma de Mel (Manual)')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig('mel_spectrogram_manual.png')
        print("Guardado 'mel_spectrogram_manual.png'")
    else:
        print("Uso: python manual_audio.py <archivo_audio.mp3/wav>")
