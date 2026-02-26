"""
HiFi-GAN: Neural vocoder for converting mel spectrograms to audio waveforms.
Architecture: Generator (transposed convolutions + multi-receptive field residual blocks)
+ Multi-Period Discriminator + Multi-Scale Discriminator.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HiFiGANConfig:
    """Hyperparameters for HiFi-GAN."""

    # Generator
    upsample_rates: tuple = (8, 8, 2, 2)  # Total: 256 = hop_length
    upsample_kernel_sizes: tuple = (16, 16, 4, 4)
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    n_mels: int = 80

    # Discriminator
    mpd_periods: tuple = (2, 3, 5, 7, 11)


class ResBlock(nn.Module):
    """Residual block with multiple dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d,
                          padding=(kernel_size * d - d) // 2),
            ))
            self.convs2.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size - 1) // 2),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = x + xt
        return x


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator: Converts mel spectrogram to audio waveform.
    Uses transposed convolutions for upsampling and residual blocks
    for multi-receptive field fusion.
    """

    def __init__(self, config: HiFiGANConfig = None):
        super().__init__()
        self.config = config or HiFiGANConfig()
        cfg = self.config

        self.pre_conv = nn.Conv1d(cfg.n_mels, cfg.upsample_initial_channel, 7, padding=3)

        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        channels = cfg.upsample_initial_channel
        for i, (u_rate, u_kernel) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            self.upsamples.append(
                nn.ConvTranspose1d(
                    channels, channels // 2,
                    kernel_size=u_kernel, stride=u_rate,
                    padding=(u_kernel - u_rate) // 2,
                )
            )
            channels = channels // 2

            # Add residual blocks with different kernel sizes
            for j, (k, d) in enumerate(zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(channels, k, d))

        self.post_conv = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram (batch, n_mels, T_mel).

        Returns:
            Audio waveform (batch, 1, T_audio).
        """
        x = self.pre_conv(mel)

        n_resblocks = len(self.config.resblock_kernel_sizes)
        for i, upsample in enumerate(self.upsamples):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)

            # Sum outputs from all resblocks at this level
            xs = None
            for j in range(n_resblocks):
                rb_idx = i * n_resblocks + j
                if xs is None:
                    xs = self.resblocks[rb_idx](x)
                else:
                    xs += self.resblocks[rb_idx](x)
            x = xs / n_resblocks

        x = F.leaky_relu(x)
        x = self.post_conv(x)
        x = torch.tanh(x)

        return x


class PeriodDiscriminator(nn.Module):
    """Sub-discriminator for a specific period (captures periodic patterns)."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        self.post_conv = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Audio waveform (batch, 1, T).

        Returns:
            Tuple (prediction, feature_maps).
        """
        feature_maps = []
        b, c, t = x.shape

        # Pad and reshape to 2D (batch, 1, T // period, period)
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)

        x = self.post_conv(x)
        feature_maps.append(x)
        x = x.flatten(1, -1)

        return x, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator: multiple sub-discriminators with different periods."""

    def __init__(self, config: HiFiGANConfig = None):
        super().__init__()
        cfg = config or HiFiGANConfig()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in cfg.mpd_periods
        ])

    def forward(self, x: torch.Tensor) -> tuple[list, list]:
        predictions = []
        feature_maps = []
        for disc in self.discriminators:
            pred, fmaps = disc(x)
            predictions.append(pred)
            feature_maps.append(fmaps)
        return predictions, feature_maps


class ScaleDiscriminator(nn.Module):
    """Sub-discriminator at a specific scale."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        self.post_conv = nn.Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        x = self.post_conv(x)
        feature_maps.append(x)
        x = x.flatten(1, -1)
        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator: operates at different audio resolutions."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x: torch.Tensor) -> tuple[list, list]:
        predictions = []
        feature_maps = []
        for pool, disc in zip(self.pools, self.discriminators):
            x_pooled = pool(x)
            pred, fmaps = disc(x_pooled)
            predictions.append(pred)
            feature_maps.append(fmaps)
        return predictions, feature_maps


def generator_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    """Generator adversarial loss (LSGAN)."""
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)
    return loss


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_gen_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """Discriminator loss (LSGAN)."""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_gen_outputs):
        loss += torch.mean((1 - dr) ** 2) + torch.mean(dg ** 2)
    return loss


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    gen_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching loss between real and generated feature maps."""
    loss = 0
    for rf_list, gf_list in zip(real_features, gen_features):
        for rf, gf in zip(rf_list, gf_list):
            loss += F.l1_loss(gf, rf)
    return loss * 2
