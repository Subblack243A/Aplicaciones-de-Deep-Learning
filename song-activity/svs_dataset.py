"""
SVS Dataset: PyTorch Datasets for Tacotron 2 and FastSpeech 2 training.
Loads prepared dataset with wavs, transcripts, pitch, energy, and durations.
"""

from __future__ import annotations
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from svs_text_processor import SVSTextProcessor
from svs_audio_processor import SVSAudioProcessor


class TacotronDataset(Dataset):
    """
    Dataset for Tacotron 2 training.
    Returns: text (token IDs), mel spectrogram, gate target.
    """

    def __init__(self, dataset_dir: str, text_processor: SVSTextProcessor = None, audio_processor: SVSAudioProcessor = None):
        """
        Args:
            dataset_dir: Path to the prepared dataset directory.
            text_processor: SVSTextProcessor instance.
            audio_processor: SVSAudioProcessor instance.
        """
        self.dataset_dir = dataset_dir
        self.text_proc = text_processor or SVSTextProcessor()
        self.audio_proc = audio_processor or SVSAudioProcessor()

        self.samples = self._load_transcripts()

    def _load_transcripts(self) -> list[tuple[str, str]]:
        """Loads (segment_name, text) pairs from transcripts.txt."""
        transcripts_path = os.path.join(self.dataset_dir, "transcripts.txt")
        samples = []
        with open(transcripts_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    samples.append((parts[0], parts[1]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        segment_name, text = self.samples[idx]
        wav_path = os.path.join(self.dataset_dir, "wavs", f"{segment_name}.wav")

        # Text → token IDs
        text_ids = self.text_proc.text_to_sequence(text)
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        # Audio → mel spectrogram
        audio = self.audio_proc.load_audio(wav_path)
        mel = self.audio_proc.get_mel_spectrogram(audio)  # (n_mels, T)
        mel_tensor = torch.from_numpy(mel).float()

        # Gate target: zeros except last frame = 1 (stop signal)
        gate = torch.zeros(mel_tensor.size(1), dtype=torch.float)
        gate[-1] = 1.0

        return text_tensor, mel_tensor, gate

    @staticmethod
    def collate_fn(batch):
        """Custom collate with dynamic padding."""
        texts, mels, gates = zip(*batch)

        text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
        mel_lengths = torch.tensor([m.size(1) for m in mels], dtype=torch.long)

        # Pad texts
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

        # Pad mels (n_mels, T) → need to pad along T dimension
        max_mel_len = max(m.size(1) for m in mels)
        n_mels = mels[0].size(0)
        padded_mels = torch.zeros(len(mels), n_mels, max_mel_len)
        for i, m in enumerate(mels):
            padded_mels[i, :, :m.size(1)] = m

        # Pad gates
        padded_gates = torch.zeros(len(gates), max_mel_len)
        for i, g in enumerate(gates):
            padded_gates[i, :g.size(0)] = g

        return padded_texts, text_lengths, padded_mels, mel_lengths, padded_gates


class FastSpeechDataset(Dataset):
    """
    Dataset for FastSpeech 2 training.
    Returns: text, mel, duration, pitch, energy.
    """

    def __init__(self, dataset_dir: str, text_processor: SVSTextProcessor = None, audio_processor: SVSAudioProcessor = None):
        """
        Args:
            dataset_dir: Path to the prepared dataset directory.
            text_processor: SVSTextProcessor instance.
            audio_processor: SVSAudioProcessor instance.
        """
        self.dataset_dir = dataset_dir
        self.text_proc = text_processor or SVSTextProcessor()
        self.audio_proc = audio_processor or SVSAudioProcessor()

        self.samples = self._load_transcripts()

    def _load_transcripts(self) -> list[tuple[str, str]]:
        transcripts_path = os.path.join(self.dataset_dir, "transcripts.txt")
        samples = []
        with open(transcripts_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    samples.append((parts[0], parts[1]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        segment_name, text = self.samples[idx]
        wav_path = os.path.join(self.dataset_dir, "wavs", f"{segment_name}.wav")

        # Text → token IDs
        text_ids = self.text_proc.text_to_sequence(text)
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        # Audio → mel spectrogram
        audio = self.audio_proc.load_audio(wav_path)
        mel = self.audio_proc.get_mel_spectrogram(audio)
        mel_tensor = torch.from_numpy(mel).float()

        # Load features
        duration = self._load_feature("durations", segment_name)
        pitch = self._load_feature("pitch", segment_name)
        energy = self._load_feature("energy", segment_name)

        return text_tensor, mel_tensor, duration, pitch, energy

    def _load_feature(self, feature_type: str, segment_name: str) -> torch.Tensor:
        """Loads a precomputed feature (.npy) as a tensor."""
        path = os.path.join(self.dataset_dir, feature_type, f"{segment_name}.npy")
        if os.path.exists(path):
            return torch.from_numpy(np.load(path)).float()
        return torch.zeros(1)

    @staticmethod
    def collate_fn(batch):
        """Custom collate with dynamic padding for all features."""
        texts, mels, durations, pitches, energies = zip(*batch)

        text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
        mel_lengths = torch.tensor([m.size(1) for m in mels], dtype=torch.long)

        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

        max_mel_len = max(m.size(1) for m in mels)
        n_mels = mels[0].size(0)
        padded_mels = torch.zeros(len(mels), n_mels, max_mel_len)
        for i, m in enumerate(mels):
            padded_mels[i, :, :m.size(1)] = m

        padded_durations = pad_sequence(durations, batch_first=True, padding_value=0)
        padded_pitches = pad_sequence(pitches, batch_first=True, padding_value=0)
        padded_energies = pad_sequence(energies, batch_first=True, padding_value=0)

        return (padded_texts, text_lengths, padded_mels, mel_lengths,
                padded_durations, padded_pitches, padded_energies)


def create_tacotron_dataloader(dataset_dir: str, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """Creates a DataLoader for Tacotron 2 training."""
    dataset = TacotronDataset(dataset_dir)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=TacotronDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def create_fastspeech_dataloader(dataset_dir: str, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """Creates a DataLoader for FastSpeech 2 training."""
    dataset = FastSpeechDataset(dataset_dir)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=FastSpeechDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
