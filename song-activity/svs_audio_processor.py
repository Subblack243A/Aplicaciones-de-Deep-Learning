"""
SVSAudioProcessor: Audio processing for Singing Voice Synthesis.
Handles mel spectrogram generation at 22050Hz with 80 mel bands
and Griffin-Lim inverse transform.
"""

from __future__ import annotations
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt


class SVSAudioProcessor:
    """
    Processes audio for SVS models: mel spectrogram generation,
    Griffin-Lim inversion, and audio I/O.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        ref_level_db: float = 20.0,
        min_level_db: float = -100.0,
    ):
        """
        Args:
            sample_rate: Audio sample rate (22050 Hz for SVS).
            n_fft: FFT window size.
            hop_length: Hop between frames.
            win_length: Window length.
            n_mels: Number of mel frequency bands.
            fmin: Minimum frequency for mel filterbank.
            fmax: Maximum frequency for mel filterbank.
            ref_level_db: Reference level for dB normalization.
            min_level_db: Minimum dB level (clips below this).
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db

    def load_audio(self, path: str) -> np.ndarray:
        """
        Loads an audio file and resamples to target sample rate.

        Args:
            path: Path to the audio file.

        Returns:
            Audio waveform as numpy array.
        """
        audio, _ = librosa.load(path, sr=self.sample_rate)
        return audio

    def get_mel_spectrogram(self, wav: np.ndarray) -> np.ndarray:
        """
        Computes a normalized mel spectrogram (80 bands, dB scale).

        Args:
            wav: Audio waveform.

        Returns:
            Mel spectrogram (n_mels, T) normalized to [0, 1] range.
        """
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max) - self.ref_level_db
        # Normalize to [0, 1]
        mel_norm = np.clip((mel_db - self.min_level_db) / (-self.min_level_db), 0, 1)
        return mel_norm.astype(np.float32)

    def mel_to_audio(self, mel_norm: np.ndarray, n_iter: int = 60) -> np.ndarray:
        """
        Converts a normalized mel spectrogram back to audio using Griffin-Lim.

        Args:
            mel_norm: Normalized mel spectrogram (n_mels, T) in [0, 1].
            n_iter: Number of Griffin-Lim iterations.

        Returns:
            Reconstructed audio waveform.
        """
        # Denormalize
        mel_db = mel_norm * (-self.min_level_db) + self.min_level_db + self.ref_level_db
        mel_power = librosa.db_to_power(mel_db)

        # Inverse mel to STFT magnitudes
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        mel_basis_pinv = np.linalg.pinv(mel_basis)
        stft_mag = np.maximum(1e-10, np.dot(mel_basis_pinv, mel_power))
        stft_mag = np.sqrt(stft_mag)

        # Griffin-Lim
        audio = librosa.griffinlim(
            stft_mag,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return audio

    def save_wav(self, audio: np.ndarray, path: str) -> str:
        """
        Saves audio to a WAV file.

        Args:
            audio: Audio waveform.
            path: Output file path.

        Returns:
            Path to the saved file.
        """
        # Normalize amplitude
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
        sf.write(path, audio, self.sample_rate)
        print(f"Audio saved to '{path}'")
        return path

    def save_mel_plot(self, mel: np.ndarray, path: str) -> str:
        """
        Saves a mel spectrogram visualization.

        Args:
            mel: Mel spectrogram (n_mels, T).
            path: Output image path.

        Returns:
            Path to the saved image.
        """
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            mel, sr=self.sample_rate, x_axis="time", y_axis="mel",
            hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax,
        )
        plt.colorbar(format="%+2.0f")
        plt.title("Mel Spectrogram (SVS)")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def get_audio_duration(self, wav: np.ndarray) -> float:
        """Returns duration in seconds."""
        return len(wav) / self.sample_rate

    def get_mel_frames(self, wav: np.ndarray) -> int:
        """Returns number of mel frames for a given audio."""
        return 1 + len(wav) // self.hop_length
