import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveRNN(nn.Module):
    """
    WaveRNN: Vocoder neuronal recurrente.
    Genera audio muestra por muestra a partir del espectrograma Mel.
    """
    def __init__(self, n_mels=80, rnn_dim=512, fc_dim=512, upsample_factors=[4, 4, 4, 4]):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.upsample = nn.ModuleList()
        for factor in upsample_factors:
            self.upsample.append(nn.Sequential(
                nn.ConvTranspose1d(n_mels if len(self.upsample) == 0 else rnn_dim,
                                   rnn_dim, kernel_size=factor * 2,
                                   stride=factor, padding=factor // 2),
                nn.LeakyReLU(0.2)
            ))
        self.rnn = nn.GRU(rnn_dim + 1, rnn_dim, batch_first=True)
        self.fc1 = nn.Linear(rnn_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 256)

    def forward(self, mel, audio_target=None):
        x = mel
        for up in self.upsample:
            x = up(x)
        x = x.transpose(1, 2)
        if audio_target is not None:
            audio_input = audio_target.unsqueeze(-1).float() / 128.0 - 1.0
            rnn_input = torch.cat([x[:, :audio_input.size(1), :], audio_input], dim=-1)
            rnn_out, _ = self.rnn(rnn_input)
            out = F.relu(self.fc1(rnn_out))
            out = self.fc2(out)
            return out
        else:
            return self._inference(x)

    @torch.no_grad()
    def _inference(self, conditioning):
        B, T, _ = conditioning.shape
        h = conditioning.new_zeros(1, B, self.rnn_dim)
        sample = conditioning.new_zeros(B, 1, 1)
        output = []
        for t in range(T):
            rnn_input = torch.cat([conditioning[:, t:t+1, :], sample], dim=-1)
            rnn_out, h = self.rnn(rnn_input, h)
            logits = self.fc2(F.relu(self.fc1(rnn_out)))
            probs = F.softmax(logits.squeeze(1), dim=-1)
            sample_idx = torch.multinomial(probs, 1)
            sample = sample_idx.unsqueeze(-1).float() / 128.0 - 1.0
            output.append(sample_idx.squeeze(-1))
        return torch.stack(output, dim=1)
