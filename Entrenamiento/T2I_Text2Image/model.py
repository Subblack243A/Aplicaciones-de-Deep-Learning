"""
T2I Text-to-Image Model — All architecture in one file.

TextEncoder  : Transformer-based encoder (Embedding + PositionalEncoding + TransformerEncoder)
ImageDecoder : ConvTranspose2d decoder with Cross-Attention at each resolution (8→16→32→64)
Text2ImageModel : Composition of encoder + decoder with .generate() for inference
"""

import math
import torch
import torch.nn as nn

# ── Defaults (importable by train.py) ─────────────────────────────────
VOCAB_SIZE = 128
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 4
NOISE_DIM = 128
IMG_SIZE = 64
MAX_TEXT_LEN = 32


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=MAX_TEXT_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_ENCODER_LAYERS, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        padding_mask = x == 0
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = self.pos_encoding(embedded)
        features = self.transformer(embedded, src_key_padding_mask=padding_mask)
        return self.layer_norm(features)


class SpatialCrossAttention(nn.Module):
    def __init__(self, spatial_dim, text_dim=D_MODEL, nhead=NHEAD):
        super().__init__()
        self.proj_in = nn.Linear(spatial_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=nhead, batch_first=True, dropout=0.1)
        self.proj_out = nn.Linear(text_dim, spatial_dim)
        self.norm = nn.LayerNorm(spatial_dim)

    def forward(self, spatial, text_features):
        B, C, H, W = spatial.shape
        residual = spatial
        x = spatial.permute(0, 2, 3, 1).reshape(B, H * W, C)
        q = self.proj_in(x)
        attn_out, _ = self.cross_attn(query=q, key=text_features, value=text_features)
        attn_out = self.proj_out(attn_out)
        attn_out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = residual + attn_out
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        return out.permute(0, 3, 1, 2)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, text_dim=D_MODEL):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = nn.GELU()
        self.cross_attn = SpatialCrossAttention(spatial_dim=out_ch, text_dim=text_dim)

    def forward(self, x, text_features):
        x = self.activation(self.bn(self.upconv(x)))
        return self.cross_attn(x, text_features)


class ImageDecoder(nn.Module):
    """noise(128) → 8×8×512 → 16×16×256 → 32×32×128 → 64×64×64 → 64×64×3"""
    def __init__(self, noise_dim=NOISE_DIM, d_model=D_MODEL, base_ch=512):
        super().__init__()
        self.base_ch = base_ch
        self.fc_noise = nn.Sequential(nn.Linear(noise_dim, 8 * 8 * base_ch), nn.GELU())
        self.block1 = DecoderBlock(base_ch, 256, d_model)
        self.block2 = DecoderBlock(256, 128, d_model)
        self.block3 = DecoderBlock(128, 64, d_model)
        self.final_conv = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, noise, text_features):
        B = noise.size(0)
        x = self.fc_noise(noise).view(B, self.base_ch, 8, 8)
        x = self.block1(x, text_features)
        x = self.block2(x, text_features)
        x = self.block3(x, text_features)
        return self.final_conv(x)


class Text2ImageModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, noise_dim=NOISE_DIM):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size=vocab_size, d_model=d_model)
        self.image_decoder = ImageDecoder(noise_dim=noise_dim, d_model=d_model)
        self.noise_dim = noise_dim

    def forward(self, text, noise):
        text_features = self.text_encoder(text)
        return self.image_decoder(noise, text_features)

    @torch.no_grad()
    def generate(self, text, seed=None, device=None):
        self.eval()
        if device:
            text = text.to(device)
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn(text.size(0), self.noise_dim, device=text.device)
        return self(text, noise)
