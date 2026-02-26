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
    parser.add_argument('--text', type=str, default="Bury all your secrets in my skin", help='Text to sing (infer mode)')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--ckpt', type=str, default='checkpoints/tacotron2_final.pt', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output.wav', help='Output audio file (infer mode)')
    
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
        print(f"Starting inference mode for text: '{args.text}'")
        sing_from_text(args.text, args.ckpt, args.output, config=config, device=device)

if __name__ == "__main__":
    main()
