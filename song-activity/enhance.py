import os
import argparse
import torchaudio
import torch

try:
    from audio_converter import AudioConverter
except ImportError:
    print("Error: No se encontró audio_converter. Asegúrate de ejecutar este script desde el directorio de song-activity.")
    exit(1)

from speechbrain.inference.separation import SepformerSeparation

class SpeechEnhancerWrapper:
    """
    Wrapper de alto nivel para extraer audio de un video y limpiarlo 
    usando nuestro pipeline de Speech Enhancement.
    """
    def __init__(self, use_finetuned: bool = False, model_path: str = "./results/se_finetuned/finetuned_separator.ckpt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SepFormer model on {self.device}...")
        
        self.model = SepformerSeparation.from_hparams(
             source="speechbrain/sepformer-wham16k",
             savedir="pretrained_models/sepformer-wham16k",
             run_opts={"device": self.device}
        )
        
        if use_finetuned and os.path.exists(model_path):
            print(f"Injecting fine-tuned weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            try:
                self.model.mods.load_state_dict(state_dict, strict=False)
                print("Fine-tuned weights correctly injected!")
            except Exception as e:
                print(f"Warning: Could not strictly load fine-tuned dict. Fallback to pre-trained. {e}")
        elif use_finetuned:
            print(f"Warning: Fine-tuned model not found at {model_path}. Fallback to pre-trained.")


    def enhance(self, input_path: str, output_path: str = None) -> str:
        """
        Flujo de inferencia completo.
        """
        print(f"\n[1/3] Extrayendo audio 16kHz de {input_path}...")
        wav_path = AudioConverter.any_to_wav(input_path, sample_rate=16000)
        
        if output_path is None:
            base_name, _ = os.path.splitext(input_path)
            output_path = f"{base_name}_enhanced.wav"
            
        print("[2/3] Procesando el Tensor de audio a través de la Red Neuronal...")
        wav_tensor, sr = torchaudio.load(wav_path)
        
        batch_wavs = wav_tensor
        if batch_wavs.dim() == 1:
             batch_wavs = batch_wavs.unsqueeze(0)
             
        batch_wavs = batch_wavs.to(self.device)
        
        with torch.no_grad():
             estimations = self.model.separate_batch(batch_wavs)
             
        enhanced_audio_tensor = estimations[0, :, 0].cpu()
        
        print(f"[3/3] Guardando audio limpio en {output_path}...")
        torchaudio.save(output_path, enhanced_audio_tensor.unsqueeze(0), 16000)
        
        if wav_path != input_path:
             os.remove(wav_path)
             
        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean/Denoise Audio Track from MP4/WAV")
    parser.add_argument("--input", type=str, required=True, help="Input Media file (MP4/MP3/WAV)")
    parser.add_argument("--output", type=str, default=None, help="Output WAV path")
    parser.add_argument("--use_finetuned", action="store_true", help="Try to use fine-tuned ckpt if available")
    
    args = parser.parse_args()
    
    enhancer = SpeechEnhancerWrapper(use_finetuned=args.use_finetuned)
    final_wav_path = enhancer.enhance(args.input, args.output)
    
    print("\n" + "="*50)
    print("Process Finished Successfully!")
    print(f"You can now run: python main.py predict --model_path [ASR_MODEL] --input {final_wav_path}")
    print("="*50)
