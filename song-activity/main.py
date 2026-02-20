import os
import sys
import argparse
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from mp4_to_mp3_converter import convert_mp4_to_mp3
from manual_audio import hpss_manual, mel_spectrogram_manual

def process_video_pipeline(video_path):
    """
    Ejecuta el pipeline completo: MP4 -> MP3 -> Separación de Voz -> Espectrograma.
    """
    print(f"--- Iniciando Procesamiento para: {video_path} ---")
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    mp3_path = f"{base_name}.mp3"
    
    print(f"[1/4] Convirtiendo video a audio ({mp3_path})...")
    convert_mp4_to_mp3(video_path, mp3_path)
    
    if not os.path.exists(mp3_path):
        print("Error: No se pudo generar el archivo MP3.")
        return
        
    print(f"[2/4] Cargando audio (16kHz) y aplicando HPSS manual...")
    y, sr = librosa.load(mp3_path, sr=16000)
    
    print(f"      -> Ejecutando HPSS (margin_h=2.5, margin_p=1.0, kernel=31)...")
    y_voice, y_drums, y_music = hpss_manual(y, sr, margin_harmonic=2.5, margin_percussive=1.0, kernel_size=31)
    
    # Guardar resultados
    voice_path = base_name + "_voice.wav"
    drums_path = base_name + "_drums.wav"
    music_path = base_name + "_music_no_voice.wav"
    
    sf.write(voice_path, y_voice, sr)
    sf.write(drums_path, y_drums, sr)
    sf.write(music_path, y_music, sr)
    
    print(f"      -> Voz guardada en: {os.path.basename(voice_path)}")
    print(f"      -> Batería (Percusión) guardada en: {os.path.basename(drums_path)}")
    print(f"      -> Música (Sin voz) guardada en: {os.path.basename(music_path)}")
    print(f"[3/4] Generando Espectrograma de Mel manual para la voz...")
    mel_spec = mel_spectrogram_manual(y_voice, sr)
    
    mel_image_filename = f"{base_name}_voice_mel.png"
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
    plt.title(f'Espectrograma Mel (Voz): {base_name}')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(mel_image_filename)
    plt.close()
    
    print(f"      -> Imagen guardada en: {mel_image_filename}")
    
    print(f"[4/4] Proceso Completado.")
    print(f"\nLISTO PARA STT: El archivo de audio limpio para el modelo de voz a texto es:\n{os.path.abspath(voice_path)}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <archivo_video.mp4>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: El archivo '{video_path}' no existe.")
        sys.exit(1)
        
    process_video_pipeline(video_path)

if __name__ == "__main__":
    main()
