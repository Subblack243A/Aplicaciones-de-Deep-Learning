import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from models.tacotron2 import Tacotron2, Tacotron2Config, Tacotron2Loss
from dataset import SingingDataset, collate_fn
from utils.audio_utils import AudioProcessor
from utils.visualizer import TrainingVisualizer
import os

# ── Detectar dispositivo ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True          # kernels más rápidos
    print(f"[GPU] {torch.cuda.get_device_name(0)} | "
        f"VRAM libre: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[CPU] No se detectó GPU, entrenando en CPU.")


def train_tacotron2(model, dataloader, val_dataloader, config, device=device):
    """Bucle de entrenamiento completo para Tacotron 2 con Mixed Precision (FP16)."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = Tacotron2Loss()
    visualizer = TrainingVisualizer()

    # GradScaler: sólo activo en GPU (en CPU es un no-op)
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    model.to(device)
    model.train()

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_postnet_loss = 0.0
        epoch_gate_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            text_padded   = batch['text'].to(device, non_blocking=True)
            text_lengths  = batch['text_lengths'].to(device, non_blocking=True)
            mel_padded    = batch['mel'].to(device, non_blocking=True)
            gate_target   = batch['gate'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward con autocast (FP16 en GPU, FP32 en CPU)
            with autocast(enabled=use_amp):
                mel_out, mel_postnet, gate_out, alignments = model(
                    text_padded, text_lengths, mel_padded
                )

                # Ajustar dimensiones si es necesario
                min_len     = min(mel_out.size(2), mel_padded.size(2))
                mel_out     = mel_out[:, :, :min_len]
                mel_postnet = mel_postnet[:, :, :min_len]
                gate_out    = gate_out[:, :min_len]
                mel_target  = mel_padded[:, :, :min_len]
                gate_tgt    = gate_target[:, :min_len]

                total_loss, mel_l, post_l, gate_l = criterion(
                    mel_out, mel_postnet, gate_out, mel_target, gate_tgt
                )

            # Backward con escalado de gradientes (evita underflow en FP16)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss         += total_loss.item()
            epoch_mel_loss     += mel_l.item()
            epoch_postnet_loss += post_l.item()
            epoch_gate_loss    += gate_l.item()
            n_batches          += 1

        scheduler.step()

        # Promedios
        avg_loss    = epoch_loss         / max(n_batches, 1)
        avg_mel     = epoch_mel_loss     / max(n_batches, 1)
        avg_postnet = epoch_postnet_loss / max(n_batches, 1)
        avg_gate    = epoch_gate_loss    / max(n_batches, 1)

        # Validación simple
        val_loss = validate_tacotron2(model, val_dataloader, criterion, device, use_amp)

        # Registro
        visualizer.log_epoch(
            epoch, avg_loss, val_loss,
            mel_loss=avg_mel,
            postnet_loss=avg_postnet,
            gate_loss=avg_gate
        )

        if epoch % 2 == 0:
            print(f"Época {epoch}/{config.epochs} | "
                f"Loss: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                f"Mel: {avg_mel:.4f} | PostNet: {avg_postnet:.4f} | "
                f"Gate: {avg_gate:.4f}")

            # Guardar checkpoint intermedio
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/tacotron2_epoch_{epoch}.pt')

    # Liberar caché de VRAM al terminar
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Generar curva de aprendizaje
    visualizer.plot_learning_curves(
        title="Tacotron 2 - Curva de Aprendizaje",
        save_path="tacotron2_learning_curve.png"
    )
    return model, visualizer


def validate_tacotron2(model, dataloader, criterion, device, use_amp=False):
    """Evalúa el modelo en el conjunto de validación."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in dataloader:
            text_padded  = batch['text'].to(device, non_blocking=True)
            text_lengths = batch['text_lengths'].to(device, non_blocking=True)
            mel_padded   = batch['mel'].to(device, non_blocking=True)
            gate_target  = batch['gate'].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                mel_out, mel_postnet, gate_out, _ = model(
                    text_padded, text_lengths, mel_padded
                )
                min_len = min(mel_out.size(2), mel_padded.size(2))
                loss, _, _, _ = criterion(
                    mel_out[:, :, :min_len], mel_postnet[:, :, :min_len],
                    gate_out[:, :min_len],   mel_padded[:, :, :min_len],
                    gate_target[:, :min_len]
                )
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / max(n_batches, 1)


if __name__ == "__main__":
    # Configuración de prueba rápida
    config    = Tacotron2Config()
    processor = AudioProcessor()

    if os.path.exists('dataset'):
        dataset = SingingDataset('dataset', processor)

        # pin_memory=True acelera las transferencias CPU→GPU
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=(device.type == 'cuda'),
            num_workers=2,
        )

        model = Tacotron2(config)
        train_tacotron2(model, train_loader, train_loader, config, device=device)
    else:
        print("Dataset not found. Please prepare the data in 'dataset/' directory.")
