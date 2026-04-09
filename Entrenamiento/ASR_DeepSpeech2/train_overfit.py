import os
import subprocess
import tensorflow as tf
import numpy as np
import librosa
from text_encoder import TextEncoder
from audio_processor import AudioProcessor
from model import ASRModel
from trainer import Trainer

# Configuración
SONGS_DIR = os.path.join("music", "LibriSpeech", "train-clean-100")
SONGS = [
    {
        "audio": "speak.mp4",
        "trans": "speak.trans.txt"
    },
    {
        "audio": "snuff.mp4",
        "trans": "snuff.trans.txt"
    }
]

SAMPLE_RATE = 16000
N_MELS = 128
EPOCHS = 500
BATCH_SIZE = 1  # Bajar batch_size para evitar OOM

class SongDataset:
    def __init__(self, sample_rate=16000, n_mels=128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.encoder = TextEncoder()
        self.processor = AudioProcessor(sample_rate=sample_rate, n_mels=n_mels)
        self.samples = []

    def separate_vocals(self, input_path):
        """Usa demucs para separar la canción y retorna la ruta de las vocales."""
        base_dir = os.path.dirname(input_path)
        out_dir = os.path.join(base_dir, "demucs_out")
        song_name = os.path.splitext(os.path.basename(input_path))[0]
        vocals_path = os.path.join(out_dir, "htdemucs", song_name, "vocals.wav")

        if not os.path.exists(vocals_path):
            print(f"Separando vocales de {os.path.basename(input_path)} usando demucs...")
            try:
                subprocess.run(
                    ["demucs", "-o", out_dir, input_path],
                    check=True
                )
            except Exception as e:
                print(f"Error al separar vocales con demucs: {e}")
                raise e
        else:
            print(f"Vocales ya separadas para {os.path.basename(input_path)}")

        return vocals_path

    def load_samples(self):
        self.samples = []
        print(f"Buscando y separando canciones en: {os.path.abspath(SONGS_DIR)}")
        for song in SONGS:
            audio_path = os.path.join(SONGS_DIR, song["audio"])
            trans_path = os.path.join(SONGS_DIR, song["trans"])

            print(f"Checking: {os.path.abspath(audio_path)}")
            if not os.path.exists(audio_path):
                print(f"ERROR: No se encontró audio {os.path.abspath(audio_path)}")
                continue

            # Separar vocales con Demucs
            try:
                vocals_path = self.separate_vocals(audio_path)
            except Exception as e:
                print(f"Omitiendo {audio_path} por error en separación.")
                continue

            if not os.path.exists(trans_path):
                print(f"ERROR: No se encontró trans {os.path.abspath(trans_path)}")
                continue

            with open(trans_path, "r", encoding="utf-8") as f:
                transcript = f.read().lower().strip()

            self.samples.append((vocals_path, transcript))
        print(f"Cargadas {len(self.samples)} pistas vocales para overfitting.")

    def _process_sample(self, audio_path, transcript):
        try:
            # Decode bytes if needed (tf.py_function passes tensors)
            if isinstance(audio_path, tf.Tensor):
                audio_path = audio_path.numpy().decode("utf-8")
                transcript = transcript.numpy().decode("utf-8")

            # print(f"Procesando: {audio_path}")
            # Cargar y procesar audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr,
                n_fft=self.processor.n_fft,
                hop_length=self.processor.hop_length,
                n_mels=self.n_mels,
                fmin=self.processor.fmin,
                fmax=self.processor.fmax,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = AudioProcessor.normalize(mel_db)
            mel_db = mel_db.T  # (time, n_mels)

            # Codificar texto
            label = self.encoder.encode(transcript)

            # TRUNCAR: Evitar OOM limitando a ~15 segundos (1500 frames)
            max_len = 1500
            if mel_db.shape[0] > max_len:
                mel_db = mel_db[:max_len, :]

            return mel_db.astype(np.float32), np.array(label, dtype=np.int32), np.int32(mel_db.shape[0]), np.int32(len(label))
        except Exception as e:
            print(f"Error procesando {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty arrays to avoid crashing the pipeline
            return np.zeros((1, self.n_mels), dtype=np.float32), np.array([self.encoder.BLANK_TOKEN], dtype=np.int32), np.int32(1), np.int32(1)

    def _tf_process_sample(self, audio_path, transcript):
        """Wrapper de _process_sample para tf.py_function"""
        spectrogram, label, input_len, label_len = tf.py_function(
            func=self._process_sample,
            inp=[audio_path, transcript],
            Tout=[tf.float32, tf.int32, tf.int32, tf.int32]
        )

        # Necesarios para padded_batch
        spectrogram.set_shape([None, self.n_mels])
        label.set_shape([None])
        input_len.set_shape([])
        label_len.set_shape([])

        # Convertir a dims [1] requeridas por trainer.py (il_array / ll_array)
        return spectrogram, label, tf.expand_dims(input_len, axis=-1), tf.expand_dims(label_len, axis=-1)

    def create_tf_dataset(self, batch_size=2):
        if not self.samples:
            return None, 0

        audio_paths = [s[0] for s in self.samples]
        transcripts = [s[1] for s in self.samples]

        # Dataset desde rutas y textos
        dataset = tf.data.Dataset.from_tensor_slices((audio_paths, transcripts))

        # Mapeo en paralelo a numpy arrays
        dataset = dataset.map(
            self._tf_process_sample,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Padded batch dinámico
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None, self.n_mels],   # Spectrogram
                [None],                # Label
                [1],                   # Input length
                [1]                    # Label length
            ),
            padding_values=(
                0.0,
                self.encoder.BLANK_TOKEN,
                0,
                0
            )
        )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        steps = int(np.ceil(len(self.samples) / batch_size))
        return dataset, steps

def main():
    # Setup de Precisión Mixta (Acelera entrenamiento en GPU)
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Precisión mixta habilitada (mixed_float16).")
    except Exception as e:
        print(f"No se pudo habilitar precisión mixta: {e}")

    # Setup de Memoria GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Usando GPU: {gpus}")
        except RuntimeError as e:
            print(e)

    # Datos
    dataset = SongDataset(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
    dataset.load_samples()
    train_ds, steps = dataset.create_tf_dataset(batch_size=BATCH_SIZE)

    # Modelo
    encoder = TextEncoder()
    # Usamos validation dataset = train dataset para ver el overfitting
    asr = ASRModel(rnn_units=512, dropout=0.0) # Dropout 0 para overfitting más rápido

    # Construir con shape dummy para init
    input_shape = (None, N_MELS)
    model = asr.build(input_shape=input_shape, vocab_size=encoder.vocab_size)
    asr.summary()

    # Entrenar
    trainer = Trainer(model=model, encoder=encoder, learning_rate=1e-3)

    # Directorio de salida
    save_path = "model_overfit"
    os.makedirs(save_path, exist_ok=True)

    print("Iniciando Overfitting...")
    trainer.train(
        train_dataset=train_ds,
        val_dataset=train_ds, # Validar con lo mismo para ver como memoriza
        epochs=EPOCHS,
        checkpoint_dir=os.path.join(save_path, "checkpoints"),
        save_best=True,
        total_batches=steps
    )

    # Guardar final
    final_path = os.path.join(save_path, "final_model_overfit.keras")
    trainer.save_model(model, final_path)
    trainer.plot_history(save_path=os.path.join(save_path, "overfit_history.png"))

if __name__ == "__main__":
    main()

