import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# MÓDULO 3: Visualización de curvas de aprendizaje
# ============================================================

class TrainingVisualizer:
    """
    Registra y grafica las métricas de entrenamiento.
    Genera las curvas de aprendizaje requeridas:
    - Loss total vs. épocas
    - Loss por componente (si aplica)
    - Comparación train vs. validation
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epoch_numbers = []
        self.component_losses = {}  # Para losses individuales

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None,
                  **component_losses):
        """Registra las métricas de una época."""
        self.epoch_numbers.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        for name, value in component_losses.items():
            if name not in self.component_losses:
                self.component_losses[name] = []
            self.component_losses[name].append(value)

    def plot_learning_curves(self, title: str = "Curva de Aprendizaje", save_path: str = None):
        """
        Genera gráfica de Número de Épocas vs. Error.
        Evidencia gráfica requerida por el proyecto.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Gráfica 1: Loss Total ---
        ax1 = axes[0]
        ax1.plot(self.epoch_numbers, self.train_losses, 'b-', linewidth=2, label='Entrenamiento', alpha=0.8)
        if self.val_losses:
            ax1.plot(self.epoch_numbers, self.val_losses, 'r--', linewidth=2, label='Validación', alpha=0.8)
        ax1.set_xlabel('Número de Época', fontsize=12)
        ax1.set_ylabel('Error (Loss)', fontsize=12)
        ax1.set_title(f'{title} - Loss Total', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Escala logarítmica para mejor visualización

        # --- Gráfica 2: Losses por Componente ---
        ax2 = axes[1]
        if self.component_losses:
            colors = plt.cm.Set2(np.linspace(0, 1, max(len(self.component_losses), 1)))
            for i, (name, values) in enumerate(self.component_losses.items()):
                ax2.plot(self.epoch_numbers[:len(values)], values, color=colors[i], linewidth=2, label=name, alpha=0.8)
        ax2.set_xlabel('Número de Época', fontsize=12)
        ax2.set_ylabel('Error (Loss)', fontsize=12)
        ax2.set_title(f'{title} - Losses por Componente', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")
        plt.close()

    def plot_spectrogram_comparison(self, original_mel, generated_mel, title="Comparación de Espectrogramas", save_path=None):
        """Compara espectrograma original vs generado."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        ax1.imshow(original_mel, aspect='auto', origin='lower', cmap='magma')
        ax1.set_title('Espectrograma Original', fontsize=12)
        ax1.set_ylabel('Banda Mel')

        ax2.imshow(generated_mel, aspect='auto', origin='lower', cmap='magma')
        ax2.set_title('Espectrograma Generado por la Red', fontsize=12)
        ax2.set_ylabel('Banda Mel')
        ax2.set_xlabel('Frame Temporal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
