import torch
import numpy as np
import os
import argparse
import soundfile as sf # soundfile es más moderno y estable que librosa.output

from models.tacotron2 import Tacotron2, Tacotron2Config
from utils.text_utils import text_to_sequence
from utils.audio_utils import AudioProcessor

def generar_audio_por_linea(texto, model, config, device, processor):
    """Genera el array numpy de audio para una sola línea de texto."""
    sequence = text_to_sequence(texto)
    # Dependiendo de tu text_to_sequence, puede que necesites .unsqueeze(0) para simular el batch_size
    text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mel_output, alignments = model.inference(text_tensor)
        mel_output = mel_output.squeeze(0).cpu().numpy()
        
    wav = processor.mel_to_audio(mel_output)
    return wav

def sing_from_text(input_source, model_path, output_path, is_file=False, config=None, device='cuda'):
    """
    Genera canto a partir de texto o archivo .txt y guarda la canción completa en un solo audio.
    """
    if config is None:
        config = Tacotron2Config()
    
    # 1. Cargar el modelo UNA SOLA VEZ para optimizar memoria
    model = Tacotron2(config)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Modelo cargado de {model_path}")
    else:
        print(f"⚠️ Advertencia: {model_path} no encontrado. Usando pesos aleatorios.")
    
    model.to(device)
    model.eval()
    processor = AudioProcessor()

    # 2. Leer las líneas (desde txt o desde texto directo)
    if is_file and os.path.exists(input_source):
        with open(input_source, 'r', encoding='utf-8') as f:
            # Quitamos líneas vacías
            lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"📖 Leyendo {len(lines)} líneas desde el archivo txt...")
    else:
        lines = [input_source.strip()]
        print("🎤 Procesando frase única...")

    # 3. Procesar línea por línea y acumular el audio
    todos_los_audios = []
    
    # Configurar un pequeño silencio entre frases (0.5 segundos)
    duracion_silencio = 0.5 
    muestras_silencio = int(duracion_silencio * config.sample_rate)
    array_silencio = np.zeros(muestras_silencio, dtype=np.float32)

    for i, line in enumerate(lines):
        print(f"⏳ Sintetizando parte {i+1}/{len(lines)}: '{line}'")
        wav_segment = generar_audio_por_linea(line, model, config, device, processor)
        todos_los_audios.append(wav_segment)
        todos_los_audios.append(array_silencio) # Agregamos pausa al final de la frase

    # 4. Concatenar y exportar la canción completa
    if todos_los_audios:
        print("🔄 Uniendo las pistas...")
        cancion_completa = np.concatenate(todos_los_audios)
        
        try:
            sf.write(output_path, cancion_completa, config.sample_rate)
            duracion_total = len(cancion_completa) / config.sample_rate
            print(f"🎉 ¡Éxito! Canción guardada en '{output_path}' (Duración: {duracion_total:.2f} segundos)")
        except Exception as e:
            print(f"❌ Error al guardar el audio: {e}")
    else:
        print("❌ No se encontró texto para procesar.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia para Canción Completa")
    parser.add_argument('--text_file', type=str, help='Ruta a tu archivo .txt con la letra completa')
    parser.add_argument('--text', type=str, default="Bury all your secrets in my skin", help='Texto directo a cantar (se ignora si usas text_file)')
    parser.add_argument('--ckpt', type=str, default='checkpoints/tacotron2_epoch_500.pt', help='Ruta del modelo')
    parser.add_argument('--output', type=str, default='cancion_completa.wav', help='Archivo de salida WAV')
    
    args = parser.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ejecutar la lógica dependiendo de si pasaste un .txt o no
    if args.text_file:
        sing_from_text(args.text_file, args.ckpt, args.output, is_file=True, device=dev)
    else:
        sing_from_text(args.text, args.ckpt, args.output, is_file=False, device=dev)