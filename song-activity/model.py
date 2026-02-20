"""
ASRModel: Arquitectura del modelo de reconocimiento automático de voz.
Basado en DeepSpeech 2: CNN → Bi-LSTM → Dense → CTC.
"""

import tensorflow as tf


class CTCLayer(tf.keras.layers.Layer):
    """
    Capa CTC personalizada que calcula la pérdida CTC durante el entrenamiento.
    Añade la pérdida al modelo automáticamente.
    """

    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        """
        Calcula y añade la pérdida CTC.

        Args:
            y_true: Etiquetas verdaderas (enteros).
            y_pred: Predicciones del modelo (probabilidades).
            input_length: Longitud de las secuencias de entrada.
            label_length: Longitud de las etiquetas.

        Returns:
            y_pred sin modificar.
        """
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


class ASRModel:
    """
    Modelo ASR (Automatic Speech Recognition) basado en DeepSpeech 2.

    Arquitectura:
        CNN: Extrae features locales del espectrograma
        Bi-LSTM: Captura dependencias temporales bidireccionales
        Dense: Proyecta features al tamaño del vocabulario
        CTC: Decodifica secuencias de features en texto
    """

    def __init__(self, rnn_units: int = 256, dropout: float = 0.1):
        """
        Args:
            rnn_units: Número de unidades en cada capa LSTM.
            dropout: Tasa de dropout en las capas LSTM.
        """
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.model = None

    def build(self, input_shape: tuple, vocab_size: int) -> tf.keras.Model:
        """
        Construye el modelo ASR.

        Args:
            input_shape: Shape de entrada (time_steps, n_mels).
            vocab_size: Tamaño del vocabulario (sin blank token).

        Returns:
            Modelo de Keras compilado.
        """
        # Input: espectrograma (time_steps, n_mels)
        inputs = tf.keras.Input(shape=input_shape, name="input_spectrogram")

        # Expandir dimensión para Conv2D: (batch, time, freq, 1)
        x = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))(inputs)

        # === CNN: extrae features locales ===
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((1, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((1, 2))(x)

        # Reshape: colapsar frecuencia y canales para la RNN
        # (batch, time, freq/4, 64) -> (batch, time, freq/4 * 64)
        # Usar tf.shape para manejar dimensiones dinámicas
        x = tf.keras.layers.Lambda(
            lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], t.shape[2] * t.shape[3]))
        )(x)

        # === Bi-LSTM: features temporales bidireccionales ===
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True, dropout=self.dropout)
        )(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True, dropout=self.dropout)
        )(x)

        # === Dense: proyección al vocabulario + blank ===
        outputs = tf.keras.layers.Dense(vocab_size + 1, activation='softmax', name="output", dtype='float32')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="asr_deepspeech2")
        return self.model

    def summary(self):
        """Imprime el resumen del modelo."""
        if self.model:
            self.model.summary()

    @staticmethod
    def ctc_loss_fn(y_true, y_pred, input_length, label_length):
        """
        Calcula la pérdida CTC.

        Args:
            y_true: Etiquetas verdaderas.
            y_pred: Predicciones del modelo.
            input_length: Longitud de la entrada.
            label_length: Longitud de la etiqueta.

        Returns:
            Valor de la pérdida CTC.
        """
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
