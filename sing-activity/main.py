import os
import argparse
import torch
from models.tacotron2 import Tacotron2, Tacotron2Config
from dataset import SingingDataset, collate_fn
from torch.utils.data import DataLoader
from utils.audio_utils import AudioProcessor
from train import train_tacotron2
from inference import sing_from_text

def main():
    parser = argparse.ArgumentParser(description="Singing Voice Synthesis using Tacotron 2")
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], default='infer', help='Mode to run: train or infer')
    
    # 🌟 NUEVO: Argumento para recibir el archivo .txt
    parser.add_argument('--text_file', type=str, help='Path to .txt file with lyrics (infer mode)')
    parser.add_argument('--text', type=str, default="Bury all your secrets in my skin", help='Text to sing (ignored if text_file is used)')
    
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--ckpt', type=str, default='checkpoints/tacotron2_final.pt', help='Path to model checkpoint')
    
    # Modificamos la ayuda del output para reflejar que será el nombre base
    parser.add_argument('--output', type=str, default='voz_salida', help='Base name for output audio files')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Tacotron2Config()
    processor = AudioProcessor()

    if args.mode == 'train':
        print("Starting training mode...")
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory '{args.data_dir}' not found.")
            return
        
        dataset = SingingDataset(args.data_dir, processor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        
        model = Tacotron2(config)
        train_tacotron2(model, dataloader, dataloader, config, device=device)
        
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        torch.save(model.state_dict(), args.ckpt)
        print(f"Training finished. Model saved at {args.ckpt}")

    elif args.mode == 'infer':
        # 🌟 NUEVO LÓGICA: Verificamos si el usuario pasó un archivo .txt
        if args.text_file and os.path.exists(args.text_file):
            print(f"📖 Leyendo la letra desde: '{args.text_file}'")
            
            with open(args.text_file, 'r', encoding='utf-8') as f:
                lineas = f.readlines()
            
            contador = 1
            for linea in lineas:
                texto_limpio = linea.strip() # Quitamos espacios extra y saltos de línea
                
                if texto_limpio: # Solo procesamos si la línea no está en blanco
                    # Crea un nombre numerado, ej: voz_salida_01.wav
                    nombre_salida = f"{args.output}_{contador:02d}.wav"
                    print(f"\n🎤 Generando línea {contador}: '{texto_limpio}'")
                    
                    sing_from_text(texto_limpio, args.ckpt, nombre_salida, config=config, device=device)
                    contador += 1
            
            print("\n✅ ¡Toda la canción ha sido procesada con éxito!")
            
        else:
            # Si no pasas un .txt, funciona como antes (una sola frase)
            print(f"Starting inference mode for text: '{args.text}'")
            nombre_salida = f"{args.output}.wav"
            sing_from_text(args.text, args.ckpt, nombre_salida, config=config, device=device)

if __name__ == "__main__":
    main()