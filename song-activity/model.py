"""
ASRModel: Automatic Speech Recognition architecture.
Based on DeepSpeech 2: CNN → Bi-LSTM → Dense → CTC.
Implemented in PyTorch with CUDA support.
"""

import torch
import torch.nn as nn


class ASRModel(nn.Module):
    """
    DeepSpeech 2 ASR model.

    Architecture:
        CNN: Extracts local features from the mel spectrogram
        Bi-LSTM: Captures bidirectional temporal dependencies
        Dense: Projects features to vocabulary size
        CTC: Decodes feature sequences into text
    """

    def __init__(self, n_mels: int = 128, vocab_size: int = 28, rnn_units: int = 256, dropout: float = 0.1):
        """
        Args:
            n_mels: Number of mel frequency bands.
            vocab_size: Vocabulary size (without blank token).
            rnn_units: Number of units per LSTM layer.
            dropout: Dropout rate for LSTM layers.
        """
        super().__init__()
        self.vocab_size = vocab_size

        # CNN: extract local features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )

        # After CNN: freq dimension = n_mels // 4, channels = 64
        rnn_input_size = (n_mels // 4) * 64

        # Bi-LSTM: bidirectional temporal features
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Dense: project to vocabulary + blank
        self.fc = nn.Linear(rnn_units * 2, vocab_size + 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram (batch, time_steps, n_mels).

        Returns:
            Log probabilities (batch, time_steps, vocab_size + 1).
        """
        # (batch, time, freq) -> (batch, 1, time, freq)
        x = x.unsqueeze(1)

        # CNN
        x = self.cnn(x)

        # (batch, channels, time, freq//4) -> (batch, time, channels * freq//4)
        batch, channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, time, channels * freq)

        # Bi-LSTM
        x, _ = self.lstm(x)

        # Dense + LogSoftmax
        x = self.fc(x)
        x = self.log_softmax(x)

        return x

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes output sequence lengths after CNN (no temporal pooling used).

        Args:
            input_lengths: Input sequence lengths.

        Returns:
            Output sequence lengths (same as input since no time pooling).
        """
        return input_lengths

    @staticmethod
    def ctc_loss_fn(
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes CTC loss.

        Args:
            log_probs: Model output (batch, time, vocab+1).
            targets: Target labels (batch, max_target_len).
            input_lengths: Input sequence lengths.
            target_lengths: Target sequence lengths.

        Returns:
            CTC loss value.
        """
        # CTC expects (time, batch, classes)
        log_probs = log_probs.permute(1, 0, 2)
        loss_fn = nn.CTCLoss(blank=log_probs.shape[-1] - 1, zero_infinity=True)
        return loss_fn(log_probs, targets, input_lengths, target_lengths)
