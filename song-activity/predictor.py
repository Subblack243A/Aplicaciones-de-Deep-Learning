"""
Predictor: Loads a trained ASR model and transcribes audio/video files.
Supports greedy and beam search decoding.
Implemented in PyTorch.
"""

import os
import numpy as np
import torch

from text_encoder import TextEncoder
from audio_processor import AudioProcessor
from audio_converter import AudioConverter
from audio_preprocessor import AudioPreprocessor
from model import ASRModel


class Predictor:
    """
    Loads a trained ASR model and generates transcriptions
    from audio or video files.
    """

    def __init__(
        self,
        model_path: str,
        encoder: TextEncoder = None,
        processor: AudioProcessor = None,
        n_mels: int = 128,
        rnn_units: int = 256,
    ):
        """
        Args:
            model_path: Path to saved model weights (.pt).
            encoder: TextEncoder instance (default: creates new one).
            processor: AudioProcessor instance (default: creates new one).
            n_mels: Number of mel bands (must match training config).
            rnn_units: RNN units (must match training config).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder or TextEncoder()
        self.processor = processor or AudioProcessor()
        self.preprocessor = AudioPreprocessor()
        self.model = self._load_model(model_path, n_mels, rnn_units)

    def _load_model(self, model_path: str, n_mels: int, rnn_units: int) -> ASRModel:
        """Loads the model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at '{model_path}'")

        model = ASRModel(
            n_mels=n_mels,
            vocab_size=self.encoder.vocab_size,
            rnn_units=rnn_units,
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model loaded from '{model_path}' on {self.device}")
        return model

    @torch.no_grad()
    def transcribe_audio(self, audio_path: str, beam_width: int = None) -> str:
        """
        Transcribes an audio file.

        Args:
            audio_path: Path to the audio file (WAV, FLAC, etc.).
            beam_width: Beam search width (None = greedy decoding).

        Returns:
            Transcribed text.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

        # Process audio → spectrogram → prepare for model
        prepared = self.processor.process_audio_file(audio_path)

        # Convert to tensor
        tensor_input = torch.from_numpy(prepared).float().to(self.device)

        # Inference
        log_probs = self.model(tensor_input)

        # Decode
        if beam_width:
            return self.encoder.decode_beam(log_probs, beam_width)
        return self.encoder.decode_greedy(log_probs, os.path.abspath(audio_path))

    def transcribe_video(self, video_path: str, beam_width: int = None) -> str:
        """
        Transcribes a video file (converts to WAV first).

        Args:
            video_path: Path to the video file.
            beam_width: Beam search width (None = greedy).

        Returns:
            Transcribed text.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: '{video_path}'")

        wav_path = AudioConverter.any_to_wav(video_path)
        vocals_path = self.preprocessor.extract_vocals(os.path.abspath(wav_path))
        try:
            return self.transcribe_audio(vocals_path, beam_width)
        finally:
            ext = os.path.splitext(video_path)[1].lower()
            if ext != '.wav' and os.path.isfile(wav_path):
                os.remove(wav_path)

    def transcribe(self, input_path: str, beam_width: int = None) -> str:
        """
        Transcribes an audio or video file (auto-detects type).

        Args:
            input_path: Path to the file.
            beam_width: Beam search width (None = greedy).

        Returns:
            Transcribed text.
        """
        ext = os.path.splitext(input_path)[1].lower()
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

        if ext in video_exts:
            return self.transcribe_video(input_path, beam_width)
        else:
            wav_path = AudioConverter.any_to_wav(input_path)
            try:
                return self.transcribe_audio(wav_path, beam_width)
            finally:
                if ext != '.wav' and os.path.isfile(wav_path) and wav_path != input_path:
                    os.remove(wav_path)

    @torch.no_grad()
    def transcribe_full(self, input_path: str, beam_width: int = None) -> dict:
        """
        Transcribes an audio/video file and returns all intermediate artifacts.

        Args:
            input_path: Path to the audio or video file.
            beam_width: Beam search width (None = greedy).

        Returns:
            Dict with keys: transcription, vocals_path, mel_spectrogram,
            audio, sample_rate.
        """
        ext = os.path.splitext(input_path)[1].lower()
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

        # Step 1: Convert to WAV if needed
        wav_path = AudioConverter.any_to_wav(input_path)

        # Step 2: Extract vocals
        vocals_path = self.preprocessor.extract_vocals(os.path.abspath(wav_path))

        # Step 3: Load audio from vocals
        audio, sr = self.processor.load_audio(vocals_path)

        # Step 4: Generate mel spectrogram
        mel_spectrogram = self.processor.to_mel_spectrogram(audio)

        # Step 5: Prepare for model
        prepared = self.processor.prepare_for_model(mel_spectrogram)
        tensor_input = torch.from_numpy(prepared).float().to(self.device)

        # Step 6: Predict
        log_probs = self.model(tensor_input)

        # Step 7: Decode
        if beam_width:
            transcription = self.encoder.decode_beam(log_probs, beam_width)
        else:
            transcription = self.encoder.decode_greedy(log_probs, os.path.abspath(vocals_path))

        # Cleanup intermediate WAV
        abs_wav = os.path.abspath(wav_path)
        abs_vocals = os.path.abspath(vocals_path)
        if ext in video_exts and abs_wav != abs_vocals and os.path.isfile(wav_path):
            os.remove(wav_path)

        return {
            "transcription": transcription,
            "vocals_path": vocals_path,
            "mel_spectrogram": mel_spectrogram,
            "audio": audio,
            "sample_rate": sr,
        }

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculates Word Error Rate.

        Args:
            reference: Reference text.
            hypothesis: Transcribed text (hypothesis).

        Returns:
            WER as float (0.0 = perfect, 1.0 = 100% error).
        """
        try:
            from jiwer import wer
            return wer(reference, hypothesis)
        except ImportError:
            print("Install 'jiwer' to calculate WER: pip install jiwer")
            return None
