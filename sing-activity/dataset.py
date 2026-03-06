import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from utils.text_utils import text_to_sequence
from utils.audio_utils import AudioProcessor

class SingingDataset(Dataset):
    """
    Dataset para SVS (Singing Voice Synthesis).
    Carga textos y audios, preprocesándolos para Tacotron 2.
    """
    def __init__(self, data_path, processor: AudioProcessor):
        self.data_path = data_path
        self.processor = processor
        self.file_list = []
        # Espera un archivo 'transcripts.txt' en data_path con formato: "audio_id|texto"
        transcripts_path = os.path.join(data_path, 'transcripts.txt')
        if os.path.exists(transcripts_path):
            with open(transcripts_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) == 2:
                        self.file_list.append(parts)
        else:
            print(f"Warning: {transcripts_path} not found.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_id, text = self.file_list[idx]
        audio_path = os.path.join(self.data_path, 'wavs', f"{audio_id}.wav")
        
        # Procesar Texto
        text_seq = text_to_sequence(text)
        text_tensor = torch.LongTensor(text_seq)
        
        # Procesar Audio
        wav = self.processor.load_audio(audio_path)
        mel = self.processor.get_mel_spectrogram(wav)
        mel_tensor = torch.FloatTensor(mel)
        
        # Gate (fin de secuencia) - todo ceros excepto el último frame
        gate_target = torch.zeros(mel_tensor.size(1))
        gate_target[-1] = 1.0
        
        return {
            'text': text_tensor,
            'mel': mel_tensor,
            'gate': gate_target,
            'text_len': len(text_seq),
            'mel_len': mel_tensor.size(1)
        }

def collate_fn(batch):
    """Collate function for dynamic padding."""
    text_lengths = torch.LongTensor([item['text_len'] for item in batch])
    mel_lengths = torch.LongTensor([item['mel_len'] for item in batch])
    
    max_text_len = text_lengths.max()
    max_mel_len = mel_lengths.max()
    n_mels = batch[0]['mel'].size(0)
    
    text_padded = torch.zeros(len(batch), max_text_len).long()
    mel_padded = torch.zeros(len(batch), n_mels, max_mel_len)
    gate_padded = torch.zeros(len(batch), max_mel_len)
    
    for i, item in enumerate(batch):
        text_padded[i, :item['text_len']] = item['text']
        mel_padded[i, :, :item['mel_len']] = item['mel']
        gate_padded[i, :item['mel_len']] = item['gate']
        
    return {
        'text': text_padded,
        'text_lengths': text_lengths,
        'mel': mel_padded,
        'gate': gate_padded,
        'mel_lengths': mel_lengths
    }
