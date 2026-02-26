"""
Tacotron 2: Sequence-to-sequence autoregressive model for Singing Voice Synthesis.
Architecture: Encoder (Embedding → Conv1D → BiLSTM) → Location-Sensitive Attention
→ Decoder (autoregressive) → PostNet (residual Conv1D).
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Tacotron2Config:
    """Hyperparameters for Tacotron 2."""

    vocab_size: int = 80
    encoder_embedding_dim: int = 512
    encoder_n_convolutions: int = 3
    encoder_kernel_size: int = 5
    encoder_lstm_units: int = 256

    n_mels: int = 80
    n_frames_per_step: int = 1

    attention_rnn_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

    decoder_rnn_dim: int = 1024
    prenet_dim: int = 256
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5

    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5

    dropout: float = 0.5


class Encoder(nn.Module):
    """Tacotron 2 Encoder: Embedding → 3× Conv1D + BN + ReLU → BiLSTM."""

    def __init__(self, config: Tacotron2Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.encoder_embedding_dim)

        convolutions = []
        for _ in range(config.encoder_n_convolutions):
            conv = nn.Sequential(
                nn.Conv1d(
                    config.encoder_embedding_dim, config.encoder_embedding_dim,
                    kernel_size=config.encoder_kernel_size,
                    padding=(config.encoder_kernel_size - 1) // 2,
                ),
                nn.BatchNorm1d(config.encoder_embedding_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            convolutions.append(conv)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            config.encoder_embedding_dim,
            config.encoder_lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: (batch, text_len) integer token IDs.
            text_lengths: (batch,) lengths.

        Returns:
            Encoder outputs (batch, text_len, encoder_dim).
        """
        x = self.embedding(text)  # (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T) for Conv1d

        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)  # (B, T, D)

        x = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x


class LocationSensitiveAttention(nn.Module):
    """Location-sensitive attention with cumulative attention weights."""

    def __init__(self, config: Tacotron2Config):
        super().__init__()
        encoder_dim = config.encoder_lstm_units * 2

        self.query_layer = nn.Linear(config.attention_rnn_dim, config.attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, config.attention_dim, bias=False)
        self.location_conv = nn.Conv1d(
            2, config.attention_location_n_filters,
            kernel_size=config.attention_location_kernel_size,
            padding=(config.attention_location_kernel_size - 1) // 2,
        )
        self.location_dense = nn.Linear(config.attention_location_n_filters, config.attention_dim, bias=False)
        self.v = nn.Linear(config.attention_dim, 1, bias=False)
        self.score_mask_value = -float("inf")

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Decoder RNN output (batch, attention_rnn_dim).
            memory: Encoder outputs (batch, text_len, encoder_dim).
            processed_memory: Pre-processed memory (batch, text_len, attention_dim).
            attention_weights_cat: Concatenated attention weights (batch, 2, text_len).
            mask: Padding mask (batch, text_len).

        Returns:
            Tuple of (context_vector, attention_weights).
        """
        processed_query = self.query_layer(query.unsqueeze(1))  # (B, 1, D)
        processed_location = self.location_dense(
            self.location_conv(attention_weights_cat).transpose(1, 2)
        )  # (B, T, D)

        energies = self.v(torch.tanh(processed_query + processed_memory + processed_location))
        energies = energies.squeeze(-1)  # (B, T)

        if mask is not None:
            energies.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(energies, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)

        return context, attention_weights


class Prenet(nn.Module):
    """Prenet: 2× Linear + ReLU + Dropout (always active, even in eval)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dropout = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dropout always active (important for Tacotron 2 autoregressive stability)
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.dropout, training=True)
        return x


class PostNet(nn.Module):
    """PostNet: 5× Conv1D residual blocks for mel spectrogram refinement."""

    def __init__(self, config: Tacotron2Config):
        super().__init__()
        channels = config.postnet_embedding_dim

        self.convolutions = nn.ModuleList()
        # First conv
        self.convolutions.append(nn.Sequential(
            nn.Conv1d(config.n_mels, channels, kernel_size=config.postnet_kernel_size,
                      padding=(config.postnet_kernel_size - 1) // 2),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Dropout(config.dropout),
        ))
        # Middle convs
        for _ in range(config.postnet_n_convolutions - 2):
            self.convolutions.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=config.postnet_kernel_size,
                          padding=(config.postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(config.dropout),
            ))
        # Last conv (no activation)
        self.convolutions.append(nn.Sequential(
            nn.Conv1d(channels, config.n_mels, kernel_size=config.postnet_kernel_size,
                      padding=(config.postnet_kernel_size - 1) // 2),
            nn.BatchNorm1d(config.n_mels),
            nn.Dropout(config.dropout),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram (batch, n_mels, time).

        Returns:
            Residual correction (batch, n_mels, time).
        """
        for conv in self.convolutions:
            x = conv(x)
        return x


class Decoder(nn.Module):
    """Tacotron 2 Decoder: Autoregressive with teacher forcing."""

    def __init__(self, config: Tacotron2Config):
        super().__init__()
        self.config = config
        encoder_dim = config.encoder_lstm_units * 2

        self.prenet = Prenet(config.n_mels, config.prenet_dim)

        self.attention_rnn = nn.LSTMCell(
            config.prenet_dim + encoder_dim, config.attention_rnn_dim,
        )
        self.attention = LocationSensitiveAttention(config)

        self.decoder_rnn = nn.LSTMCell(
            config.attention_rnn_dim + encoder_dim, config.decoder_rnn_dim,
        )

        self.linear_projection = nn.Linear(
            config.decoder_rnn_dim + encoder_dim, config.n_mels,
        )
        self.gate_layer = nn.Linear(
            config.decoder_rnn_dim + encoder_dim, 1,
        )

    def _init_states(self, memory: torch.Tensor):
        """Initialize decoder hidden states."""
        B = memory.size(0)
        T = memory.size(1)
        device = memory.device
        cfg = self.config
        encoder_dim = cfg.encoder_lstm_units * 2

        self.attention_rnn_h = torch.zeros(B, cfg.attention_rnn_dim, device=device)
        self.attention_rnn_c = torch.zeros(B, cfg.attention_rnn_dim, device=device)
        self.decoder_rnn_h = torch.zeros(B, cfg.decoder_rnn_dim, device=device)
        self.decoder_rnn_c = torch.zeros(B, cfg.decoder_rnn_dim, device=device)

        self.attention_weights = torch.zeros(B, T, device=device)
        self.attention_weights_cum = torch.zeros(B, T, device=device)
        self.context = torch.zeros(B, encoder_dim, device=device)

        self.processed_memory = self.attention.memory_layer(memory)
        self.memory = memory

    def _decode_step(self, decoder_input: torch.Tensor, mask: torch.Tensor = None):
        """Process a single decoder step."""
        prenet_output = self.prenet(decoder_input)

        cell_input = torch.cat([prenet_output, self.context], dim=-1)
        self.attention_rnn_h, self.attention_rnn_c = self.attention_rnn(
            cell_input, (self.attention_rnn_h, self.attention_rnn_c),
        )

        attention_weights_cat = torch.stack(
            [self.attention_weights, self.attention_weights_cum], dim=1,
        )
        self.context, self.attention_weights = self.attention(
            self.attention_rnn_h,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            mask,
        )
        self.attention_weights_cum += self.attention_weights

        decoder_input_rnn = torch.cat([self.attention_rnn_h, self.context], dim=-1)
        self.decoder_rnn_h, self.decoder_rnn_c = self.decoder_rnn(
            decoder_input_rnn, (self.decoder_rnn_h, self.decoder_rnn_c),
        )

        decoder_hidden_context = torch.cat([self.decoder_rnn_h, self.context], dim=-1)
        mel_output = self.linear_projection(decoder_hidden_context)
        gate_output = self.gate_layer(decoder_hidden_context)

        return mel_output, gate_output, self.attention_weights

    def forward(
        self,
        memory: torch.Tensor,
        mel_target: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing forward pass.

        Args:
            memory: Encoder outputs (batch, text_len, encoder_dim).
            mel_target: Target mel (batch, n_mels, mel_len).
            text_lengths: Text lengths for masking.

        Returns:
            Tuple (mel_outputs, gate_outputs, alignments).
        """
        self._init_states(memory)

        # Prepare decoder inputs (shift right, prepend zeros)
        decoder_input = torch.zeros(memory.size(0), self.config.n_mels, device=memory.device)
        mel_inputs = torch.cat([decoder_input.unsqueeze(2), mel_target], dim=2)  # (B, n_mels, T+1)

        # Create mask for attention
        max_len = memory.size(1)
        mask = torch.arange(max_len, device=memory.device).unsqueeze(0) >= text_lengths.unsqueeze(1)

        mel_outputs, gate_outputs, alignments = [], [], []

        for t in range(mel_target.size(2)):
            mel_input = mel_inputs[:, :, t]
            mel_output, gate_output, alignment = self._decode_step(mel_input, mask)
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output.squeeze(-1))
            alignments.append(alignment)

        mel_outputs = torch.stack(mel_outputs, dim=2)  # (B, n_mels, T)
        gate_outputs = torch.stack(gate_outputs, dim=1)  # (B, T)
        alignments = torch.stack(alignments, dim=1)  # (B, T_mel, T_text)

        return mel_outputs, gate_outputs, alignments

    @torch.no_grad()
    def inference(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive inference (no teacher forcing).

        Args:
            memory: Encoder outputs (batch, text_len, encoder_dim).

        Returns:
            Tuple (mel_outputs, gate_outputs, alignments).
        """
        self._init_states(memory)
        decoder_input = torch.zeros(memory.size(0), self.config.n_mels, device=memory.device)

        mel_outputs, gate_outputs, alignments = [], [], []

        for _ in range(self.config.max_decoder_steps):
            mel_output, gate_output, alignment = self._decode_step(decoder_input)
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output.squeeze(-1))
            alignments.append(alignment)

            if torch.sigmoid(gate_output).item() > self.config.gate_threshold:
                break

            decoder_input = mel_output

        mel_outputs = torch.stack(mel_outputs, dim=2)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    """Full Tacotron 2 model: Encoder → Attention → Decoder → PostNet."""

    def __init__(self, config: Tacotron2Config = None):
        super().__init__()
        self.config = config or Tacotron2Config()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.postnet = PostNet(self.config)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_target: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass with teacher forcing.

        Args:
            text: (batch, text_len) token IDs.
            text_lengths: (batch,) text lengths.
            mel_target: (batch, n_mels, mel_len) target mel spectrogram.
            mel_lengths: (batch,) mel lengths.

        Returns:
            Tuple (mel_before_postnet, mel_after_postnet, gate_outputs, alignments).
        """
        encoder_outputs = self.encoder(text, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_target, text_lengths,
        )
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    @torch.no_grad()
    def inference(self, text: torch.Tensor, text_lengths: torch.Tensor) -> dict:
        """
        Inference: generate mel spectrogram from text.

        Args:
            text: (batch, text_len) token IDs.
            text_lengths: (batch,) text lengths.

        Returns:
            Dict with mel_spectrogram, gate_outputs, alignments.
        """
        self.eval()
        encoder_outputs = self.encoder(text, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        return {
            "mel_spectrogram": mel_outputs_postnet,
            "gate_outputs": gate_outputs,
            "alignments": alignments,
        }


class Tacotron2Loss(nn.Module):
    """Combined loss: MSE(mel_before) + MSE(mel_after_postnet) + BCE(gate)."""

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_postnet_pred: torch.Tensor,
        gate_pred: torch.Tensor,
        mel_target: torch.Tensor,
        gate_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            mel_pred: Mel before PostNet (batch, n_mels, T).
            mel_postnet_pred: Mel after PostNet (batch, n_mels, T).
            gate_pred: Gate predictions (batch, T).
            mel_target: Target mel (batch, n_mels, T).
            gate_target: Target gate (batch, T).

        Returns:
            Tuple (total_loss, component_losses_dict).
        """
        mel_loss = F.mse_loss(mel_pred, mel_target)
        postnet_loss = F.mse_loss(mel_postnet_pred, mel_target)
        gate_loss = F.binary_cross_entropy_with_logits(gate_pred, gate_target)

        total = mel_loss + postnet_loss + gate_loss

        return total, {
            "mel_loss": mel_loss.item(),
            "postnet_loss": postnet_loss.item(),
            "gate_loss": gate_loss.item(),
        }
