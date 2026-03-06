import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.text_utils import symbols

# --- Hiperparámetros ---
class Tacotron2Config:
    # Texto
    n_symbols = len(symbols)       # Tamaño del vocabulario
    embedding_dim = 512            # Dimensión del embedding de caracteres

    # Encoder
    encoder_n_convolutions = 3     # Número de capas conv en el encoder
    encoder_kernel_size = 5        # Tamaño del kernel convolucional
    encoder_embedding_dim = 512    # Dimensión de salida del encoder

    # Atención
    attention_rnn_dim = 1024       # Dimensión del RNN de atención
    attention_dim = 128            # Dimensión de la capa de atención
    attention_location_n_filters = 32   # Filtros para atención location-aware
    attention_location_kernel_size = 31 # Kernel para filtros de ubicación

    # Decoder
    n_mel_channels = 80            # Bandas Mel (coincide con AudioProcessor)
    decoder_rnn_dim = 1024         # Dimensión del LSTM del decoder
    prenet_dim = 256               # Dimensión del prenet (red previa al decoder)
    max_decoder_steps = 1000       # Máximo de frames a generar
    gate_threshold = 0.5           # Umbral para detener la generación
    n_frames_per_step = 1          # Frames generados por paso

    # PostNet
    postnet_n_convolutions = 5     # Capas conv del PostNet
    postnet_embedding_dim = 512    # Dimensión del PostNet
    postnet_kernel_size = 5        # Kernel del PostNet

    # Entrenamiento
    learning_rate = 1e-3
    batch_size = 4
    epochs = 500
    weight_decay = 1e-6


# --- Componentes del Encoder ---

class ConvBlock(nn.Module):
    """Bloque convolucional 1D con BatchNorm y ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, self.training)
        return x


class Encoder(nn.Module):
    """Encoder de Tacotron 2: Embedding → 3x Conv1D → BiLSTM."""
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.n_symbols, config.embedding_dim)
        convolutions = []
        for i in range(config.encoder_n_convolutions):
            in_ch = config.embedding_dim if i == 0 else config.encoder_embedding_dim
            convolutions.append(
                ConvBlock(in_ch, config.encoder_embedding_dim,
                        config.encoder_kernel_size,
                        padding=(config.encoder_kernel_size - 1) // 2)
            )
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(
            config.encoder_embedding_dim,
            config.encoder_embedding_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, text_sequences, input_lengths):
        x = self.embedding(text_sequences).transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs


# --- Mecanismo de Atención ---

class LocationLayer(nn.Module):
    """Capa de ubicación para atención location-sensitive."""
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        self.location_conv = nn.Conv1d(
            2, attention_n_filters, attention_kernel_size,
            padding=(attention_kernel_size - 1) // 2, bias=False
        )
        self.location_dense = nn.Linear(attention_n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat):
        processed = self.location_conv(attention_weights_cat).transpose(1, 2)
        processed = self.location_dense(processed)
        return processed


class LocationSensitiveAttention(nn.Module):
    """Atención Location-Sensitive."""
    def __init__(self, attention_rnn_dim, encoder_dim, attention_dim,
                attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters, attention_location_kernel_size, attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_memory + processed_attention))
        return energies.squeeze(-1)

    def forward(self, decoder_state, memory, processed_memory,
                attention_weights_cat, mask=None):
        alignment = self.get_alignment_energies(decoder_state, processed_memory, attention_weights_cat)
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return attention_context, attention_weights


# --- PreNet, PostNet, Decoder ---

class Prenet(nn.Module):
    """PreNet del Decoder."""
    def __init__(self, in_dim, sizes=[256, 256]):
        super().__init__()
        layers = []
        prev = in_dim
        for size in sizes:
            layers.append(nn.Linear(prev, size))
            prev = size
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=0.5, training=True)
        return x


class PostNet(nn.Module):
    """PostNet de 5 capas Conv1D."""
    def __init__(self, config):
        super().__init__()
        channels = config.postnet_embedding_dim
        kernel = config.postnet_kernel_size
        padding = (kernel - 1) // 2
        self.convolutions = nn.ModuleList()
        self.convolutions.append(nn.Sequential(
            nn.Conv1d(config.n_mel_channels, channels, kernel, padding=padding),
            nn.BatchNorm1d(channels)
        ))
        for _ in range(config.postnet_n_convolutions - 2):
            self.convolutions.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel, padding=padding),
                nn.BatchNorm1d(channels)
            ))
        self.convolutions.append(nn.Sequential(
            nn.Conv1d(channels, config.n_mel_channels, kernel, padding=padding),
            nn.BatchNorm1d(config.n_mel_channels)
        ))

    def forward(self, x):
        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, self.training)
            else:
                x = F.dropout(conv(x), 0.5, self.training)
        return x


class Decoder(nn.Module):
    """Decoder autoregresivo de Tacotron 2."""
    def __init__(self, config):
        super().__init__()
        self.n_mel_channels = config.n_mel_channels
        self.encoder_dim = config.encoder_embedding_dim
        self.attention_rnn_dim = config.attention_rnn_dim
        self.decoder_rnn_dim = config.decoder_rnn_dim
        self.prenet_dim = config.prenet_dim
        self.max_decoder_steps = config.max_decoder_steps
        self.gate_threshold = config.gate_threshold

        self.prenet = Prenet(config.n_mel_channels, [config.prenet_dim, config.prenet_dim])
        self.attention_rnn = nn.LSTMCell(config.prenet_dim + config.encoder_embedding_dim, config.attention_rnn_dim)
        self.attention = LocationSensitiveAttention(
            config.attention_rnn_dim, config.encoder_embedding_dim,
            config.attention_dim, config.attention_location_n_filters,
            config.attention_location_kernel_size
        )
        self.decoder_rnn = nn.LSTMCell(config.attention_rnn_dim + config.encoder_embedding_dim, config.decoder_rnn_dim)
        self.mel_projection = nn.Linear(config.decoder_rnn_dim + config.encoder_embedding_dim, config.n_mel_channels)
        self.gate_layer = nn.Linear(config.decoder_rnn_dim + config.encoder_embedding_dim, 1)

    def initialize_decoder_states(self, memory):
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        self.attention_hidden = memory.new_zeros(B, self.attention_rnn_dim)
        self.attention_cell = memory.new_zeros(B, self.attention_rnn_dim)
        self.decoder_hidden = memory.new_zeros(B, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(B, self.decoder_rnn_dim)
        self.attention_weights = memory.new_zeros(B, MAX_TIME)
        self.attention_weights_cum = memory.new_zeros(B, MAX_TIME)
        self.attention_context = memory.new_zeros(B, self.encoder_dim)
        self.processed_memory = self.attention.memory_layer(memory)
        self.memory = memory

    def decode_step(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, 0.1, self.training)

        attention_weights_cat = torch.stack([self.attention_weights, self.attention_weights_cum], dim=1)
        self.attention_context, self.attention_weights = self.attention(
            self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat
        )
        self.attention_weights_cum += self.attention_weights

        decoder_input_rnn = torch.cat((self.attention_hidden, self.attention_context), dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input_rnn, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, 0.1, self.training)

        decoder_hidden_context = torch.cat((self.decoder_hidden, self.attention_context), dim=-1)
        mel_output = self.mel_projection(decoder_hidden_context)
        gate_output = self.gate_layer(decoder_hidden_context)
        return mel_output, gate_output, self.attention_weights

    def forward(self, memory, mel_targets, memory_lengths=None):
        go_frame = memory.new_zeros(memory.size(0), self.n_mel_channels)
        mel_targets = mel_targets.transpose(1, 2)
        decoder_inputs = torch.cat([go_frame.unsqueeze(1), mel_targets[:, :-1, :]], dim=1)
        self.initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = [], [], []
        for t in range(decoder_inputs.size(1)):
            decoder_input = self.prenet(decoder_inputs[:, t, :])
            mel_out, gate_out, attn_weights = self.decode_step(decoder_input)
            mel_outputs.append(mel_out)
            gate_outputs.append(gate_out.squeeze(-1))
            alignments.append(attn_weights)
        mel_outputs = torch.stack(mel_outputs, dim=1).transpose(1, 2)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        return mel_outputs, gate_outputs, alignments

    @torch.no_grad()
    def inference(self, memory):
        self.initialize_decoder_states(memory)
        decoder_input = memory.new_zeros(1, self.n_mel_channels)
        mel_outputs, alignments = [], []
        for _ in range(self.max_decoder_steps):
            decoder_input = self.prenet(decoder_input)
            mel_out, gate_out, attn_weights = self.decode_step(decoder_input)
            mel_outputs.append(mel_out)
            alignments.append(attn_weights)
            if torch.sigmoid(gate_out) > self.gate_threshold: break
            decoder_input = mel_out
        mel_outputs = torch.stack(mel_outputs, dim=1).transpose(1, 2)
        alignments = torch.stack(alignments, dim=1)
        return mel_outputs, alignments


class Tacotron2(nn.Module):
    """Modelo Tacotron 2 completo."""
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.postnet = PostNet(config)

    def forward(self, text_padded, text_lengths, mel_padded):
        encoder_outputs = self.encoder(text_padded, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mel_padded, text_lengths)
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    @torch.no_grad()
    def inference(self, text_sequence):
        text_padded = text_sequence.unsqueeze(0)
        text_lengths = torch.tensor([text_sequence.size(0)], device=text_sequence.device)
        encoder_outputs = self.encoder(text_padded, text_lengths)
        mel_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        return mel_outputs_postnet, alignments


class Tacotron2Loss(nn.Module):
    """Loss compuesto: MSE Mel + MSE Mel-PostNet + BCE Gate."""
    def forward(self, mel_out, mel_postnet, gate_out, mel_target, gate_target):
        mel_loss = F.mse_loss(mel_out, mel_target)
        mel_postnet_loss = F.mse_loss(mel_postnet, mel_target)
        gate_loss = F.binary_cross_entropy_with_logits(gate_out, gate_target)
        total_loss = mel_loss + mel_postnet_loss + gate_loss
        return total_loss, mel_loss, mel_postnet_loss, gate_loss
