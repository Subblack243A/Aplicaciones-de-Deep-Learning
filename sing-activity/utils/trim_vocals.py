import librosa
import soundfile as sf
import numpy as np
import os

def trim_and_remove_silence(input_path, output_path, trim_at_sec=260, top_db=30):
    """
    Corta el audio en trim_at_sec (por defecto 4:20 = 260s)
    y elimina los silencios del resto.
    """
    if not os.path.exists(input_path):
        print(f"Error: No se encontró el archivo {input_path}")
        return

    print(f"Cargando {input_path}...")
    # Cargamos el audio (por defecto librosa lo carga a 22050 Hz si no se especifica sr)
    # Usaremos sr=22050 para mantener consistencia con el proyecto
    y, sr = librosa.load(input_path, sr=22050)
    
    # 1. Cortar en el minuto 4:20 (260 segundos)
    print(f"Cortando a los {trim_at_sec} segundos...")
    num_samples_trim = int(trim_at_sec * sr)
    y_trimmed = y[:num_samples_trim]
    
    # 2. Eliminar silencios
    print(f"Eliminando silencios (umbral: {top_db} dB)...")
    # librosa.effects.split devuelve los intervalos de sonido no silencioso
    intervals = librosa.effects.split(y_trimmed, top_db=top_db)
    
    # Concatenar los intervalos no silenciosos
    y_clean = np.concatenate([y_trimmed[start:end] for start, end in intervals])
    
    print(f"Duración original: {len(y)/sr:.2f}s")
    print(f"Duración tras corte y limpieza: {len(y_clean)/sr:.2f}s")
    
    # Guardar el resultado
    print(f"Guardando en {output_path}...")
    sf.write(output_path, y_clean, sr, subtype='PCM_16')
    print("¡Listo!")

if __name__ == "__main__":
    # Rutas absolutas para evitar errores
    INPUT_WAV = "/home/subblack/uni/ADL/Aplicaciones-de-Deep-Learning/sing-activity/vocals.wav"
    OUTPUT_WAV = "/home/subblack/uni/ADL/Aplicaciones-de-Deep-Learning/sing-activity/vocals_trimmed_clean.wav"
    
    trim_and_remove_silence(INPUT_WAV, OUTPUT_WAV)
