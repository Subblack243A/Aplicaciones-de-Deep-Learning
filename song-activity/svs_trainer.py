"""
SVS Trainer: Training loops for Tacotron 2 and FastSpeech 2.
Supports CUDA acceleration, mixed precision, and training visualization.
"""

from __future__ import annotations
import os
import time
import torch
import torch.nn as nn
import numpy as np

from svs_visualizer import TrainingVisualizer


class SVSTrainer:
    """
    Unified training loop for SVS models (Tacotron 2 and FastSpeech 2).
    Handles CUDA, mixed precision, checkpoints, and visualization.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        model_name: str = "SVS",
        output_dir: str = "./training_output",
    ):
        """
        Args:
            model: SVS model (Tacotron2 or FastSpeech2).
            loss_fn: Loss function module.
            learning_rate: Learning rate.
            model_name: Name for logging and visualization.
            output_dir: Directory for saving outputs.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10,
        )

        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.visualizer = TrainingVisualizer(model_name, output_dir)

    def train_tacotron2(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 500,
        grad_clip: float = 1.0,
    ) -> None:
        """
        Training loop for Tacotron 2.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader (optional).
            epochs: Number of epochs.
            grad_clip: Max gradient norm.
        """
        best_val_loss = float("inf")
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"

        print(f"\n{'='*60}")
        print(f"  Tacotron 2 Training on {device_name}: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = []
            component_totals = {}
            start = time.time()

            for batch_idx, batch in enumerate(train_loader, 1):
                texts, text_lengths, mels, mel_lengths, gates = batch
                texts = texts.to(self.device)
                text_lengths = text_lengths.to(self.device)
                mels = mels.to(self.device)
                mel_lengths = mel_lengths.to(self.device)
                gates = gates.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    mel_pred, mel_postnet_pred, gate_pred, alignments = self.model(
                        texts, text_lengths, mels, mel_lengths,
                    )
                    # Trim predictions to match target length
                    min_len = min(mel_pred.size(2), mels.size(2))
                    loss, components = self.loss_fn(
                        mel_pred[:, :, :min_len],
                        mel_postnet_pred[:, :, :min_len],
                        gate_pred[:, :min_len],
                        mels[:, :, :min_len],
                        gates[:, :min_len],
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())
                for k, v in components.items():
                    component_totals[k] = component_totals.get(k, 0) + v

            # Epoch stats
            train_loss = np.mean(epoch_losses)
            n_batches = len(epoch_losses)
            avg_components = {k: v / n_batches for k, v in component_totals.items()}
            elapsed = time.time() - start

            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_tacotron2(val_loader)

            # Log and print
            self.visualizer.log_epoch(
                epoch, train_loss, val_loss,
                lr=self.optimizer.param_groups[0]["lr"],
                **avg_components,
            )

            val_str = f" | Val: {val_loss:.4f}" if val_loss else ""
            comp_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_components.items())
            print(f"  Epoch {epoch}/{epochs} | Loss: {train_loss:.4f}{val_str} | {comp_str} | {elapsed:.1f}s")

            # Save best
            if val_loss and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, "best_model.pt")
                print(f"    ✓ Best model (val={val_loss:.4f})")

            # Periodic checkpoint and plots
            if epoch % 50 == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pt")
                self.visualizer.plot_learning_curves()

            # LR schedule
            self.scheduler.step(val_loss if val_loss else train_loss)

        # Final outputs
        self._save_checkpoint(epochs, "final_model.pt")
        self.visualizer.plot_learning_curves()
        self.visualizer.generate_training_report()
        self.visualizer.save_metrics_json()

    @torch.no_grad()
    def _validate_tacotron2(self, val_loader) -> float:
        """Validation step for Tacotron 2."""
        self.model.eval()
        losses = []
        for batch in val_loader:
            texts, text_lengths, mels, mel_lengths, gates = batch
            texts = texts.to(self.device)
            text_lengths = text_lengths.to(self.device)
            mels = mels.to(self.device)
            gates = gates.to(self.device)

            mel_pred, mel_postnet_pred, gate_pred, _ = self.model(
                texts, text_lengths, mels, mel_lengths.to(self.device),
            )
            min_len = min(mel_pred.size(2), mels.size(2))
            loss, _ = self.loss_fn(
                mel_pred[:, :, :min_len], mel_postnet_pred[:, :, :min_len],
                gate_pred[:, :min_len], mels[:, :, :min_len], gates[:, :min_len],
            )
            losses.append(loss.item())
        return np.mean(losses)

    def train_fastspeech2(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 600,
        grad_clip: float = 1.0,
    ) -> None:
        """
        Training loop for FastSpeech 2.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader (optional).
            epochs: Number of epochs.
            grad_clip: Max gradient norm.
        """
        best_val_loss = float("inf")
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"

        print(f"\n{'='*60}")
        print(f"  FastSpeech 2 Training on {device_name}: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = []
            component_totals = {}
            start = time.time()

            for batch in train_loader:
                texts, text_lengths, mels, mel_lengths, durations, pitches, energies = batch
                texts = texts.to(self.device)
                text_lengths = text_lengths.to(self.device)
                mels = mels.to(self.device)
                durations = durations.to(self.device)
                pitches = pitches.to(self.device)
                energies = energies.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    mel_pred, predictions = self.model(
                        texts, text_lengths, durations, pitches, energies,
                    )
                    min_len = min(mel_pred.size(2), mels.size(2))
                    min_text = min(predictions["log_duration_pred"].size(1), durations.size(1))
                    min_mel = min(predictions["pitch_pred"].size(1), pitches.size(1))

                    loss, components = self.loss_fn(
                        mel_pred[:, :, :min_len], mels[:, :, :min_len],
                        predictions["log_duration_pred"][:, :min_text], durations[:, :min_text],
                        predictions["pitch_pred"][:, :min_mel], pitches[:, :min_mel],
                        predictions["energy_pred"][:, :min_mel], energies[:, :min_mel],
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())
                for k, v in components.items():
                    component_totals[k] = component_totals.get(k, 0) + v

            # Epoch stats
            train_loss = np.mean(epoch_losses)
            n_batches = len(epoch_losses)
            avg_components = {k: v / n_batches for k, v in component_totals.items()}
            elapsed = time.time() - start

            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_fastspeech2(val_loader)

            self.visualizer.log_epoch(
                epoch, train_loss, val_loss,
                lr=self.optimizer.param_groups[0]["lr"],
                **avg_components,
            )

            val_str = f" | Val: {val_loss:.4f}" if val_loss else ""
            print(f"  Epoch {epoch}/{epochs} | Loss: {train_loss:.4f}{val_str} | {elapsed:.1f}s")

            if val_loss and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, "best_model.pt")

            if epoch % 50 == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pt")
                self.visualizer.plot_learning_curves()

            self.scheduler.step(val_loss if val_loss else train_loss)

        self._save_checkpoint(epochs, "final_model.pt")
        self.visualizer.plot_learning_curves()
        self.visualizer.generate_training_report()
        self.visualizer.save_metrics_json()

    @torch.no_grad()
    def _validate_fastspeech2(self, val_loader) -> float:
        """Validation step for FastSpeech 2."""
        self.model.eval()
        losses = []
        for batch in val_loader:
            texts, text_lengths, mels, mel_lengths, durations, pitches, energies = batch
            texts = texts.to(self.device)
            text_lengths = text_lengths.to(self.device)
            mels = mels.to(self.device)
            durations = durations.to(self.device)
            pitches = pitches.to(self.device)
            energies = energies.to(self.device)

            mel_pred, predictions = self.model(
                texts, text_lengths, durations, pitches, energies,
            )
            min_len = min(mel_pred.size(2), mels.size(2))
            min_text = min(predictions["log_duration_pred"].size(1), durations.size(1))
            min_mel = min(predictions["pitch_pred"].size(1), pitches.size(1))

            loss, _ = self.loss_fn(
                mel_pred[:, :, :min_len], mels[:, :, :min_len],
                predictions["log_duration_pred"][:, :min_text], durations[:, :min_text],
                predictions["pitch_pred"][:, :min_mel], pitches[:, :min_mel],
                predictions["energy_pred"][:, :min_mel], energies[:, :min_mel],
            )
            losses.append(loss.item())
        return np.mean(losses)

    def _save_checkpoint(self, epoch: int, filename: str) -> None:
        """Saves model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
