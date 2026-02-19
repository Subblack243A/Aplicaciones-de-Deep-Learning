"""
Trainer: Loop de entrenamiento personalizado con CTC loss,
métricas, gráficas de progreso y guardado/carga de modelos.
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from text_encoder import TextEncoder


class Trainer:
    """
    Entrena el modelo ASR con CTC loss usando un loop personalizado
    con tf.GradientTape.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        encoder: TextEncoder,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            model: Modelo de Keras (ASR).
            encoder: Instancia de TextEncoder.
            learning_rate: Tasa de aprendizaje.
        """
        self.model = model
        self.encoder = encoder
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Historial de entrenamiento
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    @tf.function(reduce_retracing=True)
    def _train_step(self, batch):
        """
        Ejecuta un paso de entrenamiento.

        Args:
            batch: Tupla (spectrograms, labels, input_lengths, label_lengths).

        Returns:
            Pérdida CTC del batch.
        """
        spectrograms, labels, input_lengths, label_lengths = batch

        with tf.GradientTape() as tape:
            y_pred = self.model(spectrograms, training=True)
            # Calcular la longitud real de la salida del modelo
            # (la CNN puede reducir la dimensión temporal)
            pred_length = tf.shape(y_pred)[1]
            # Ajustar input_lengths para reflejar la salida real
            input_lens = tf.minimum(
                tf.squeeze(input_lengths, axis=-1),
                tf.cast(tf.fill([tf.shape(spectrograms)[0]], pred_length), tf.int32)
            )
            label_lens = tf.squeeze(label_lengths, axis=-1)

            loss = tf.keras.backend.ctc_batch_cost(
                labels, y_pred,
                tf.cast(tf.reshape(input_lens, (-1, 1)), tf.int32),
                tf.cast(tf.reshape(label_lens, (-1, 1)), tf.int32),
            )
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Clip gradients para evitar gradientes explosivos
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss

    @tf.function(reduce_retracing=True)
    def _val_step(self, batch):
        """
        Ejecuta un paso de validación.

        Args:
            batch: Tupla (spectrograms, labels, input_lengths, label_lengths).

        Returns:
            Pérdida CTC del batch.
        """
        spectrograms, labels, input_lengths, label_lengths = batch

        y_pred = self.model(spectrograms, training=False)
        pred_length = tf.shape(y_pred)[1]
        input_lens = tf.minimum(
            tf.squeeze(input_lengths, axis=-1),
            tf.cast(tf.fill([tf.shape(spectrograms)[0]], pred_length), tf.int32)
        )
        label_lens = tf.squeeze(label_lengths, axis=-1)

        loss = tf.keras.backend.ctc_batch_cost(
            labels, y_pred,
            tf.cast(tf.reshape(input_lens, (-1, 1)), tf.int32),
            tf.cast(tf.reshape(label_lens, (-1, 1)), tf.int32),
        )
        return tf.reduce_mean(loss)

    def _sample_prediction(self, dataset):
        """
        Genera una predicción de ejemplo del dataset para monitorear el progreso.

        Args:
            dataset: tf.data.Dataset de validación.
        """
        for batch in dataset.take(1):
            spectrograms, labels, input_lengths, label_lengths = batch
            y_pred = self.model(spectrograms[:1], training=False)

            # Decodificar predicción con CTC greedy
            input_len = np.array([y_pred.shape[1]])
            decoded, _ = tf.keras.backend.ctc_decode(
                y_pred, input_len, greedy=True
            )
            decoded_indices = tf.cast(decoded[0], dtype=tf.int32).numpy()[0]
            # Filtrar valores -1 (padding de CTC decode)
            decoded_indices = [idx for idx in decoded_indices if idx >= 0]
            pred_text = self.encoder.decode(decoded_indices)

            # Decodificar etiqueta real
            label = labels[0].numpy()
            label = [int(l) for l in label if l != self.encoder.BLANK_TOKEN]
            true_text = self.encoder.decode(label)

            print(f"  Referencia:  '{true_text}'")
            print(f"  Predicción:  '{pred_text}'")

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset = None,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoints",
        save_best: bool = True,
    ):
        """
        Entrena el modelo.

        Args:
            train_dataset: Dataset de entrenamiento.
            val_dataset: Dataset de validación (opcional).
            epochs: Número de épocas.
            checkpoint_dir: Directorio para guardar checkpoints.
            save_best: Si guardar el mejor modelo según val_loss.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = float("inf")

        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento - {epochs} épocas")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            start_time = time.time()

            # === Entrenamiento ===
            train_losses = []
            for batch_idx, batch in enumerate(train_dataset):
                loss = self._train_step(batch)
                train_losses.append(float(loss))

                if (batch_idx + 1) % 10 == 0:
                    avg_loss = np.mean(train_losses[-10:])
                    print(f"  Época {epoch+1}/{epochs} - Batch {batch_idx+1} - Loss: {avg_loss:.4f}")

            epoch_train_loss = np.mean(train_losses)
            self.history["train_loss"].append(epoch_train_loss)

            # === Validación ===
            epoch_val_loss = None
            if val_dataset is not None:
                val_losses = []
                for batch in val_dataset:
                    loss = self._val_step(batch)
                    val_losses.append(float(loss))
                epoch_val_loss = np.mean(val_losses)
                self.history["val_loss"].append(epoch_val_loss)

            elapsed = time.time() - start_time

            # Imprimir resumen de la época
            msg = (
                f"Época {epoch+1}/{epochs} - "
                f"Train Loss: {epoch_train_loss:.4f}"
            )
            if epoch_val_loss is not None:
                msg += f" - Val Loss: {epoch_val_loss:.4f}"
            msg += f" - Tiempo: {elapsed:.1f}s"
            print(msg)

            # Mostrar predicción de ejemplo cada 5 épocas
            if val_dataset is not None and (epoch + 1) % 5 == 0:
                print("  --- Predicción de ejemplo ---")
                self._sample_prediction(val_dataset)

            # Guardar mejor modelo
            if save_best and epoch_val_loss is not None:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_path = os.path.join(checkpoint_dir, "best_model.keras")
                    self.model.save(best_path)
                    print(f"  ✓ Mejor modelo guardado (val_loss: {best_val_loss:.4f})")

            # Guardar checkpoint periódico
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.keras")
                self.model.save(ckpt_path)
                print(f"  ✓ Checkpoint guardado: {ckpt_path}")

        print(f"\n{'='*60}")
        print(f"Entrenamiento finalizado")
        print(f"{'='*60}\n")

    def plot_history(self, save_path: str = None):
        """
        Grafica la historia de entrenamiento (loss por época).

        Args:
            save_path: Ruta opcional para guardar la gráfica.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(self.history["train_loss"]) + 1)
        ax.plot(epochs, self.history["train_loss"], "b-o", label="Train Loss", markersize=3)

        if self.history["val_loss"]:
            ax.plot(epochs, self.history["val_loss"], "r-o", label="Val Loss", markersize=3)

        ax.set_xlabel("Época")
        ax.set_ylabel("CTC Loss")
        ax.set_title("Historial de Entrenamiento ASR")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Gráfica guardada en '{save_path}'")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def save_model(model: tf.keras.Model, path: str):
        """
        Guarda el modelo en disco.

        Args:
            model: Modelo a guardar.
            path: Ruta de destino.
        """
        model.save(path)
        print(f"Modelo guardado en '{path}'")

    @staticmethod
    def load_model(path: str) -> tf.keras.Model:
        """
        Carga un modelo desde disco.

        Args:
            path: Ruta del modelo guardado.

        Returns:
            Modelo de Keras cargado.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado en '{path}'")
        model = tf.keras.models.load_model(path)
        print(f"Modelo cargado desde '{path}'")
        return model
