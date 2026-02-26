"""
FastSpeech 2: Non-autoregressive Transformer-based model for Singing Voice Synthesis.
Architecture: FFT Encoder → Variance Adaptor (Duration + Pitch + Energy) → FFT Decoder → Mel Linear.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FastSpeech2Config:
    """Hyperparameters for FastSpeech 2."""

    vocab_size: int = 80
    encoder_hidden: int = 256
    encoder_layers: int = 4
    encoder_heads: int = 2
    encoder_ff_dim: int = 1024

    decoder_hidden: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 2
    decoder_ff_dim: int = 1024

    n_mels: int = 80
    max_seq_len: int = 2048
    dropout: float = 0.2

    # Variance predictor
    variance_predictor_filter: int = 256
    variance_predictor_kernel: int = 3
    variance_predictor_dropout: float = 0.5

    # Pitch/energy quantization
    n_bins: int = 256
    pitch_min: float = 0.0
    pitch_max: float = 800.0
    energy_min: float = 0.0
    energy_max: float = 100.0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FFTBlock(nn.Module):
    """Feed-Forward Transformer block: Self-Attention → LayerNorm → FFN → LayerNorm."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention + residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual
        x = self.norm2(x + self.ffn(x))
        return x


class VariancePredictor(nn.Module):
    """Predicts duration, pitch, or energy: Conv1D × 2 → Linear."""

    def __init__(self, d_model: int, filter_size: int = 256, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(d_model, filter_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(filter_size, filter_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(filter_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model).

        Returns:
            Predictions (batch, seq_len).
        """
        x = x.transpose(1, 2)  # (B, D, T) for Conv1d
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return self.linear(x).squeeze(-1)


class LengthRegulator(nn.Module):
    """Expands encoder sequence according to predicted/ground-truth durations."""

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoder output (batch, text_len, d_model).
            durations: Duration per token (batch, text_len) as integers.

        Returns:
            Expanded sequence (batch, mel_len, d_model).
        """
        outputs = []
        for i in range(x.size(0)):
            expanded = torch.repeat_interleave(x[i], durations[i].long(), dim=0)
            outputs.append(expanded)

        # Pad to same length
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(len(outputs), max_len, x.size(2), device=x.device)
        for i, o in enumerate(outputs):
            padded[i, :o.size(0)] = o

        return padded


class VarianceAdaptor(nn.Module):
    """Combines Duration, Pitch, and Energy predictors with Length Regulator."""

    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        d = config.encoder_hidden

        self.duration_predictor = VariancePredictor(
            d, config.variance_predictor_filter,
            config.variance_predictor_kernel, config.variance_predictor_dropout,
        )
        self.pitch_predictor = VariancePredictor(
            d, config.variance_predictor_filter,
            config.variance_predictor_kernel, config.variance_predictor_dropout,
        )
        self.energy_predictor = VariancePredictor(
            d, config.variance_predictor_filter,
            config.variance_predictor_kernel, config.variance_predictor_dropout,
        )

        self.length_regulator = LengthRegulator()

        # Pitch and energy embeddings (quantized bins)
        self.pitch_bins = nn.Parameter(
            torch.linspace(config.pitch_min, config.pitch_max, config.n_bins - 1),
            requires_grad=False,
        )
        self.pitch_embedding = nn.Embedding(config.n_bins, d)

        self.energy_bins = nn.Parameter(
            torch.linspace(config.energy_min, config.energy_max, config.n_bins - 1),
            requires_grad=False,
        )
        self.energy_embedding = nn.Embedding(config.n_bins, d)

    def forward(
        self,
        x: torch.Tensor,
        duration_target: torch.Tensor = None,
        pitch_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: Encoder output (batch, text_len, d_model).
            duration_target: Ground truth durations (training only).
            pitch_target: Ground truth pitch (training only).
            energy_target: Ground truth energy (training only).

        Returns:
            Tuple (adapted_output, predictions_dict).
        """
        log_duration_pred = self.duration_predictor(x)

        if duration_target is not None:
            # Training: use ground truth durations
            x = self.length_regulator(x, duration_target)
        else:
            # Inference: use predicted durations
            duration_pred = torch.clamp(torch.round(torch.exp(log_duration_pred) - 1), min=0).int()
            x = self.length_regulator(x, duration_pred)

        # Pitch
        pitch_pred = self.pitch_predictor(x)
        if pitch_target is not None:
            pitch_quantized = torch.bucketize(pitch_target, self.pitch_bins)
            x = x + self.pitch_embedding(pitch_quantized)
        else:
            pitch_quantized = torch.bucketize(pitch_pred, self.pitch_bins)
            x = x + self.pitch_embedding(pitch_quantized)

        # Energy
        energy_pred = self.energy_predictor(x)
        if energy_target is not None:
            energy_quantized = torch.bucketize(energy_target, self.energy_bins)
            x = x + self.energy_embedding(energy_quantized)
        else:
            energy_quantized = torch.bucketize(energy_pred, self.energy_bins)
            x = x + self.energy_embedding(energy_quantized)

        predictions = {
            "log_duration_pred": log_duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
        }

        return x, predictions


class FastSpeech2(nn.Module):
    """
    Full FastSpeech 2 model: Encoder → Variance Adaptor → Decoder → Mel Linear.
    Non-autoregressive: generates entire mel spectrogram in one forward pass.
    """

    def __init__(self, config: FastSpeech2Config = None):
        super().__init__()
        self.config = config or FastSpeech2Config()
        d = self.config.encoder_hidden

        # Encoder
        self.embedding = nn.Embedding(self.config.vocab_size, d)
        self.pos_encoder = PositionalEncoding(d, self.config.max_seq_len, self.config.dropout)
        self.encoder_blocks = nn.ModuleList([
            FFTBlock(d, self.config.encoder_heads, self.config.encoder_ff_dim, self.config.dropout)
            for _ in range(self.config.encoder_layers)
        ])

        # Variance Adaptor
        self.variance_adaptor = VarianceAdaptor(self.config)

        # Decoder
        self.pos_decoder = PositionalEncoding(d, self.config.max_seq_len, self.config.dropout)
        self.decoder_blocks = nn.ModuleList([
            FFTBlock(d, self.config.decoder_heads, self.config.decoder_ff_dim, self.config.dropout)
            for _ in range(self.config.decoder_layers)
        ])

        # Mel output
        self.mel_linear = nn.Linear(d, self.config.n_mels)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        duration_target: torch.Tensor = None,
        pitch_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            text: (batch, text_len) token IDs.
            text_lengths: (batch,) text lengths.
            duration_target: Ground truth durations (training).
            pitch_target: Ground truth pitch per frame (training).
            energy_target: Ground truth energy per frame (training).

        Returns:
            Tuple (mel_output, predictions_dict).
        """
        # Encoder
        x = self.embedding(text)
        x = self.pos_encoder(x)

        # Create padding mask
        max_len = text.size(1)
        mask = torch.arange(max_len, device=text.device).unsqueeze(0) >= text_lengths.unsqueeze(1)

        for block in self.encoder_blocks:
            x = block(x, mask)

        # Variance Adaptor
        x, predictions = self.variance_adaptor(
            x, duration_target, pitch_target, energy_target,
        )

        # Decoder
        x = self.pos_decoder(x)
        for block in self.decoder_blocks:
            x = block(x)

        # Mel output
        mel_output = self.mel_linear(x)  # (B, T_mel, n_mels)
        mel_output = mel_output.transpose(1, 2)  # (B, n_mels, T_mel)

        return mel_output, predictions

    @torch.no_grad()
    def inference(self, text: torch.Tensor, text_lengths: torch.Tensor) -> dict:
        """
        Inference (no ground truth).

        Args:
            text: (batch, text_len) token IDs.
            text_lengths: (batch,) text lengths.

        Returns:
            Dict with mel_spectrogram and variance predictions.
        """
        self.eval()
        mel_output, predictions = self.forward(text, text_lengths)
        return {
            "mel_spectrogram": mel_output,
            **predictions,
        }


class FastSpeech2Loss(nn.Module):
    """Combined loss: MSE(mel) + MSE(log_duration) + MSE(pitch) + MSE(energy)."""

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        log_duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        pitch_pred: torch.Tensor,
        pitch_target: torch.Tensor,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            Tuple (total_loss, component_losses_dict).
        """
        mel_loss = F.mse_loss(mel_pred, mel_target)

        log_duration_target = torch.log(duration_target.float() + 1)
        duration_loss = F.mse_loss(log_duration_pred, log_duration_target)

        pitch_loss = F.mse_loss(pitch_pred, pitch_target)
        energy_loss = F.mse_loss(energy_pred, energy_target)

        total = mel_loss + duration_loss + pitch_loss + energy_loss

        return total, {
            "mel_loss": mel_loss.item(),
            "duration_loss": duration_loss.item(),
            "pitch_loss": pitch_loss.item(),
            "energy_loss": energy_loss.item(),
        }
