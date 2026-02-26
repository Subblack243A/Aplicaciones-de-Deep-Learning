"""
SVSVisualizer: Training visualization and reporting for SVS models.
Generates learning curves, spectrogram comparisons, and training reports.
"""

from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class TrainingVisualizer:
    """
    Tracks training metrics and generates visualizations
    for SVS model training progress.
    """

    def __init__(self, model_name: str = "SVS", output_dir: str = "./training_output"):
        """
        Args:
            model_name: Name of the model being trained.
            output_dir: Directory for saving plots and reports.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.component_losses: dict[str, list[float]] = {}
        self.learning_rates: list[float] = []
        self.start_time = datetime.now()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = None,
        lr: float = None,
        **component_losses: float,
    ) -> None:
        """
        Logs metrics for a training epoch.

        Args:
            epoch: Epoch number.
            train_loss: Training loss value.
            val_loss: Validation loss value (optional).
            lr: Current learning rate (optional).
            **component_losses: Named component losses (e.g., mel_loss=0.5, gate_loss=0.1).
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)

        for name, value in component_losses.items():
            if name not in self.component_losses:
                self.component_losses[name] = []
            self.component_losses[name].append(value)

    def plot_learning_curves(self, title: str = None, save_path: str = None) -> str:
        """
        Generates multi-panel learning curve plots.

        Args:
            title: Plot title.
            save_path: Path to save the plot.

        Returns:
            Path to the saved plot.
        """
        n_components = len(self.component_losses)
        has_lr = len(self.learning_rates) > 0
        n_panels = 1 + (1 if n_components > 0 else 0) + (1 if has_lr else 0)

        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(title or f"{self.model_name} Training Progress", fontsize=14, fontweight="bold")

        # Panel 1: Total Loss
        ax = axes[0]
        ax.plot(self.epochs, self.train_losses, label="Train Loss", color="#2980b9", linewidth=2)
        if self.val_losses:
            ax.plot(self.epochs[:len(self.val_losses)], self.val_losses,
                    label="Val Loss", color="#e74c3c", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        panel_idx = 1

        # Panel 2: Component Losses
        if n_components > 0:
            ax = axes[panel_idx]
            colors = plt.cm.Set2(np.linspace(0, 1, n_components))
            for (name, values), color in zip(self.component_losses.items(), colors):
                ax.plot(self.epochs[:len(values)], values, label=name, color=color, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Component Losses")
            ax.legend()
            ax.grid(True, alpha=0.3)
            panel_idx += 1

        # Panel 3: Learning Rate
        if has_lr:
            ax = axes[panel_idx]
            ax.plot(self.epochs[:len(self.learning_rates)], self.learning_rates,
                    color="#27ae60", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "learning_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Learning curves saved to '{save_path}'")
        return save_path

    def plot_spectrogram_comparison(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        save_path: str = None,
        title: str = "Spectrogram Comparison",
    ) -> str:
        """
        Plots side-by-side original vs generated spectrograms.

        Args:
            original: Original mel spectrogram (n_mels, T).
            generated: Generated mel spectrogram (n_mels, T).
            save_path: Path to save the plot.
            title: Plot title.

        Returns:
            Path to the saved plot.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        axes[0].imshow(original, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title("Original (Target)")
        axes[0].set_ylabel("Mel Band")

        axes[1].imshow(generated, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title("Generated (Model Output)")
        axes[1].set_ylabel("Mel Band")
        axes[1].set_xlabel("Time Frame")

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "spectrogram_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Spectrogram comparison saved to '{save_path}'")
        return save_path

    def generate_training_report(self, save_path: str = None) -> str:
        """
        Generates a markdown training report.

        Args:
            save_path: Path to save the report.

        Returns:
            Path to the saved report.
        """
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_report.md")

        best_train = min(self.train_losses) if self.train_losses else float("inf")
        best_val = min(self.val_losses) if self.val_losses else float("inf")
        final_train = self.train_losses[-1] if self.train_losses else float("inf")
        final_val = self.val_losses[-1] if self.val_losses else float("inf")

        report = f"""# Training Report: {self.model_name}

**Date**: {self.start_time.strftime('%Y-%m-%d %H:%M')}
**Duration**: {hours}h {minutes}m {seconds}s
**Total Epochs**: {len(self.epochs)}

## Loss Summary

| Metric | Best | Final |
|--------|------|-------|
| Train Loss | {best_train:.4f} | {final_train:.4f} |
| Val Loss | {best_val:.4f} | {final_val:.4f} |

"""
        if self.component_losses:
            report += "## Component Losses (Final)\n\n"
            report += "| Component | Value |\n|-----------|-------|\n"
            for name, values in self.component_losses.items():
                report += f"| {name} | {values[-1]:.4f} |\n"
            report += "\n"

        report += f"## Training Curves\n\n![Learning Curves](learning_curves.png)\n"

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Training report saved to '{save_path}'")
        return save_path

    def save_metrics_json(self, save_path: str = None) -> str:
        """
        Saves all metrics as a JSON file for programmatic access.

        Args:
            save_path: Path to save JSON.

        Returns:
            Path to saved JSON.
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "metrics.json")

        data = {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "component_losses": self.component_losses,
            "learning_rates": self.learning_rates,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        return save_path
