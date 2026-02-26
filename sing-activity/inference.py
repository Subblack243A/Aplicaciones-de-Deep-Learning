import torch
import numpy as np
import librosa
from models.tacotron2 import Tacotron2, Tacotron2Config
from utils.text_utils import text_to_sequence
from utils.audio_utils import AudioProcessor
import os

def sing_from_text(text, model_path, output_path, config=None, device='cuda'):
    """
    Genera canto a partir de texto y guarda el audio.
    """
    if config is None:
        config = Tacotron2Config()
    
    # Cargar modelo
    model = Tacotron2(config)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modelo cargado de {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using randomly initialized model.")
    
    model.to(device)
    model.eval()
    
    # Preprocesar texto
    sequence = text_to_sequence(text)
    text_tensor = torch.LongTensor(sequence).to(device)
    
    # Inferencia
    with torch.no_grad():
        mel_output, alignments = model.inference(text_tensor)
        mel_output = mel_output.squeeze(0).cpu().numpy()
    
    print(f"Mel espectrograma generado: {mel_output.shape}")
    
    # Reconstrucción de audio (Griffin-Lim por defecto)
    processor = AudioProcessor()
    wav = processor.mel_to_audio(mel_output)
    
    # Guardar audio
    librosa.output.write_wav(output_path, wav, sr=config.sample_rate) # Note: librosa.output.write_wav is deprecated in newer versions
    # Use soundfile instead if librosa fails
    try:
        import soundfile as sf
        sf.write(output_path, wav, config.sample_rate)
        print(f"Audio guardado en {output_path}")
    except ImportError:
        import scipy.io.wavfile as wavfile
        wavfile.write(output_path, config.sample_rate, (wav * 32767).astype(np.int16))
        print(f"Audio guardado en {output_path} (usando scipy)")

if __name__ == "__main__":
    # Ejemplo de uso
    lyrics = "Bury all your secrets in my skin"
    model_ckpt = "checkpoints/tacotron2_epoch_500.pt"
    output_audio = "generated_singing.wav"
    
    # Check if we should use CPU if CUDA is not available
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sing_from_text(lyrics, model_ckpt, output_audio, device=dev)
