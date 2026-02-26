"""
LibriSpeechDataset: Downloads, loads, and preprocesses the LibriSpeech dataset
for ASR model training.
Implemented with PyTorch Dataset and DataLoader.
"""

from __future__ import annotations
import os
import tarfile
import urllib.request
import glob
import numpy as np
import librosa

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from text_encoder import TextEncoder
from audio_processor import AudioProcessor

LIBRISPEECH_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
}


class LibriSpeechDataset(Dataset):
    """
    Loads and preprocesses the LibriSpeech dataset for ASR training.
    Inherits from torch.utils.data.Dataset for integration with DataLoader.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train-clean-100",
        sample_rate: int = 16000,
        n_mels: int = 128,
        max_audio_len: int = None,
        max_label_len: int = None,
        max_samples: int = None,
    ):
        """
        Args:
            root_dir: Root directory for the dataset.
            split: Dataset split (train-clean-100, dev-clean, test-clean).
            sample_rate: Audio sample rate.
            n_mels: Number of mel bands.
            max_audio_len: Maximum audio length in time steps.
            max_label_len: Maximum label length in characters.
            max_samples: Maximum number of samples to load.
        """
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_audio_len = max_audio_len
        self.max_label_len = max_label_len
        self.max_samples = max_samples

        self.encoder = TextEncoder()
        self.processor = AudioProcessor(sample_rate=sample_rate, n_mels=n_mels)
        self.samples: list[tuple[str, str]] = []
        self._processed: list[tuple[np.ndarray, list[int]]] = []

    def download(self) -> str:
        """
        Downloads and extracts the LibriSpeech split if it doesn't exist.

        Returns:
            Path to the extracted dataset directory.
        """
        if self.split not in LIBRISPEECH_URLS:
            raise ValueError(f"Split '{self.split}' not available. Options: {list(LIBRISPEECH_URLS.keys())}")

        os.makedirs(self.root_dir, exist_ok=True)
        dataset_path = os.path.join(self.root_dir, "LibriSpeech", self.split)

        if os.path.exists(dataset_path):
            print(f"  Dataset '{self.split}' already exists at '{dataset_path}'")
            return dataset_path

        url = LIBRISPEECH_URLS[self.split]
        tar_path = os.path.join(self.root_dir, f"{self.split}.tar.gz")

        print(f"  Downloading '{self.split}'... (this may take a while)")
        urllib.request.urlretrieve(url, tar_path, reporthook=self._download_progress)
        print()

        print(f"  Extracting '{tar_path}'...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(self.root_dir)

        if os.path.isfile(tar_path):
            os.remove(tar_path)

        return dataset_path

    @staticmethod
    def _download_progress(count, block_size, total_size) -> None:
        """Callback to show download progress."""
        pct = count * block_size * 100 // total_size
        print(f"\r  Progress: {pct}%", end="", flush=True)

    def load_samples(self) -> None:
        """
        Loads (audio_path, transcript) pairs from the dataset.
        """
        dataset_path = os.path.join(self.root_dir, "LibriSpeech", self.split)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}'. Run download() first.")

        trans_files = glob.glob(os.path.join(dataset_path, "**", "*.trans.txt"), recursive=True)
        all_samples = []

        for tf_path in trans_files:
            base_dir = os.path.dirname(tf_path)
            with open(tf_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) < 2:
                        continue
                    utt_id, transcript = parts
                    audio_path = os.path.join(base_dir, f"{utt_id}.flac")
                    if os.path.isfile(audio_path):
                        all_samples.append((audio_path, transcript.lower()))

        if self.max_samples:
            all_samples = all_samples[:self.max_samples]

        self.samples = all_samples
        print(f"  Loaded {len(self.samples)} samples from '{self.split}'")

        # Pre-process all samples
        self._processed = []
        skipped = 0
        for audio_path, transcript in self.samples:
            result = self._process_sample(audio_path, transcript)
            if result is not None:
                self._processed.append(result)
            else:
                skipped += 1

        print(f"  Processed {len(self._processed)} samples ({skipped} skipped)")

    def _process_sample(self, audio_path: str, transcript: str):
        """
        Processes a single sample: audio → spectrogram, text → integers.

        Args:
            audio_path: Path to the audio file.
            transcript: Text transcription.

        Returns:
            Tuple (spectrogram, label) or None on error.
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            mel = self.processor.to_mel_spectrogram(audio)
            mel_norm = AudioProcessor.normalize(mel)
            mel_transposed = mel_norm.T  # (time, freq)

            label = self.encoder.encode(transcript)

            # Apply length filters
            if self.max_audio_len and mel_transposed.shape[0] > self.max_audio_len:
                return None
            if self.max_label_len and len(label) > self.max_label_len:
                return None
            if len(label) == 0:
                return None

            return (mel_transposed, label)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self._processed)

    def __getitem__(self, idx: int):
        mel, label = self._processed[idx]
        return (
            torch.from_numpy(mel).float(),
            torch.tensor(label, dtype=torch.long),
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for dynamic padding.

        Returns:
            Tuple (padded_specs, padded_labels, input_lengths, label_lengths).
        """
        specs, labels = zip(*batch)

        input_lengths = torch.tensor([s.shape[0] for s in specs], dtype=torch.long)
        label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

        padded_specs = pad_sequence(specs, batch_first=True, padding_value=0.0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

        return padded_specs, padded_labels, input_lengths, label_lengths

    def create_dataloader(self, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """
        Creates a DataLoader with dynamic padding.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            num_workers: Number of worker processes.

        Returns:
            PyTorch DataLoader.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
