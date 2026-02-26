"""
Trainer: Custom training loop with CTC loss,
metrics, progress plots, and model save/load.
Implemented in PyTorch with CUDA support.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from text_encoder import TextEncoder


class Trainer:
    """
    Trains the ASR model with CTC loss using a custom PyTorch loop.
    Supports CUDA acceleration and mixed precision training.
    """

    def __init__(
        self,
        model: nn.Module,
        encoder: TextEncoder,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            model: PyTorch ASR model.
            encoder: TextEncoder instance.
            learning_rate: Learning rate for Adam optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _train_step(self, batch: tuple) -> float:
        """
        Executes a single training step.

        Args:
            batch: Tuple (spectrograms, labels, input_lengths, label_lengths).

        Returns:
            CTC loss for this batch.
        """
        specs, labels, input_lengths, label_lengths = batch
        specs = specs.to(self.device)
        labels = labels.to(self.device)
        input_lengths = input_lengths.to(self.device)
        label_lengths = label_lengths.to(self.device)

        self.optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            log_probs = self.model(specs)
            output_lengths = self.model.get_output_lengths(input_lengths)
            loss = self.model.ctc_loss_fn(log_probs, labels, output_lengths, label_lengths)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def _val_step(self, batch: tuple) -> float:
        """
        Executes a single validation step.

        Args:
            batch: Tuple (spectrograms, labels, input_lengths, label_lengths).

        Returns:
            CTC loss for this batch.
        """
        specs, labels, input_lengths, label_lengths = batch
        specs = specs.to(self.device)
        labels = labels.to(self.device)
        input_lengths = input_lengths.to(self.device)
        label_lengths = label_lengths.to(self.device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            log_probs = self.model(specs)
            output_lengths = self.model.get_output_lengths(input_lengths)
            loss = self.model.ctc_loss_fn(log_probs, labels, output_lengths, label_lengths)

        return loss.item()

    @torch.no_grad()
    def _sample_prediction(self, dataloader) -> None:
        """
        Generates a sample prediction to monitor training progress.

        Args:
            dataloader: Validation DataLoader.
        """
        self.model.eval()
        batch = next(iter(dataloader))
        specs, labels, input_lengths, label_lengths = batch
        specs = specs.to(self.device)

        log_probs = self.model(specs)
        pred_text = self.encoder.decode_greedy(log_probs[0:1])

        label_ints = labels[0, :label_lengths[0]].tolist()
        true_text = self.encoder.decode(label_ints)

        print(f"    Target:     '{true_text}'")
        print(f"    Prediction: '{pred_text}'")
        self.model.train()

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoints",
        save_best: bool = True,
        total_batches: int = None,
    ):
        """
        Trains the model.

        Args:
            train_dataloader: Training DataLoader.
            val_dataloader: Validation DataLoader (optional).
            epochs: Number of training epochs.
            checkpoint_dir: Directory for saving checkpoints.
            save_best: Whether to save the best model.
            total_batches: Total batches for progress display.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = float("inf")
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"

        print(f"\n  Training on {device_name}: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = []
            start = time.time()

            for batch_idx, batch in enumerate(train_dataloader, 1):
                loss = self._train_step(batch)
                epoch_losses.append(loss)

                if batch_idx % 10 == 0:
                    avg = np.mean(epoch_losses[-10:])
                    elapsed = time.time() - start
                    print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx} | "
                          f"Loss: {avg:.4f} | Time: {elapsed:.1f}s", end="\r")

            train_loss = np.mean(epoch_losses)
            self.train_losses.append(train_loss)
            elapsed = time.time() - start

            # Validation
            val_loss = None
            if val_dataloader:
                self.model.eval()
                val_losses = []
                for batch in val_dataloader:
                    vl = self._val_step(batch)
                    val_losses.append(vl)
                val_loss = np.mean(val_losses)
                self.val_losses.append(val_loss)

            # Print epoch summary
            val_str = f" | Val Loss: {val_loss:.4f}" if val_loss else ""
            print(f"  Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}{val_str} | {elapsed:.1f}s")

            # Sample prediction
            if val_dataloader and epoch % 5 == 0:
                self._sample_prediction(val_dataloader)

            # Save best model
            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"    ✓ Best model saved (val_loss={val_loss:.4f})")

            # Periodic checkpoint
            if epoch % 10 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), ckpt_path)

    def plot_history(self, save_path: str = None) -> None:
        """
        Plots the training history (loss per epoch).

        Args:
            save_path: Optional path to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss", color="#2980b9", linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label="Val Loss", color="#e74c3c", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("CTC Loss")
        plt.title("ASR Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Training history saved to '{save_path}'")
        plt.close()

    @staticmethod
    def save_model(model: nn.Module, path: str) -> None:
        """
        Saves the model weights to disk.

        Args:
            model: Model to save.
            path: Destination path (.pt).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Model saved to '{path}'")

    @staticmethod
    def load_model(model: nn.Module, path: str) -> nn.Module:
        """
        Loads model weights from disk.

        Args:
            model: Model instance to load weights into.
            path: Path to saved weights (.pt).

        Returns:
            Model with loaded weights.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at '{path}'")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Model loaded from '{path}'")
        return model
