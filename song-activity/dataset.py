"""
LibriSpeechDataset: Descarga, carga y preprocesamiento del dataset LibriSpeech
para entrenamiento del modelo ASR.
"""

import os
import tarfile
import urllib.request
import glob
import numpy as np
import tensorflow as tf
import librosa

from text_encoder import TextEncoder
from audio_processor import AudioProcessor


# URLs de descarga de LibriSpeech
LIBRISPEECH_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
}


class LibriSpeechDataset:
    """
    Carga y preprocesa el dataset LibriSpeech para entrenamiento ASR.

    El dataset se organiza en carpetas:
    LibriSpeech/<split>/<speaker_id>/<chapter_id>/<utterance_id>.flac
    Con archivos de transcripción:
    LibriSpeech/<split>/<speaker_id>/<chapter_id>/<speaker_id>-<chapter_id>.trans.txt
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train-clean-100",
        sample_rate: int = 16000,
        n_mels: int = 128,
        max_audio_len: int = None,
        max_label_len: int = None,
        max_samples: int = None,
    ):
        """
        Args:
            root_dir: Directorio raíz donde se descarga/almacena el dataset.
            split: Split del dataset (train-clean-100, dev-clean, test-clean).
            sample_rate: Tasa de muestreo.
            n_mels: Número de bandas mel.
            max_audio_len: Longitud máxima de audio en time steps (None = sin límite).
            max_label_len: Longitud máxima de etiquetas en caracteres (None = sin límite).
            max_samples: Número máximo de muestras a cargar (None = todas).
        """
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_audio_len = max_audio_len
        self.max_label_len = max_label_len
        self.max_samples = max_samples

        self.encoder = TextEncoder()
        self.processor = AudioProcessor(
            sample_rate=sample_rate, n_mels=n_mels
        )

        self.samples = []  # Lista de (audio_path, transcript)

    def download(self) -> str:
        """
        Descarga y extrae el split de LibriSpeech si no existe.

        Returns:
            Ruta al directorio del dataset extraído.
        """
        dataset_dir = os.path.join(self.root_dir, "LibriSpeech", self.split)
        if os.path.isdir(dataset_dir):
            print(f"Dataset ya existe en '{dataset_dir}'")
            return dataset_dir

        if self.split not in LIBRISPEECH_URLS:
            raise ValueError(
                f"Split '{self.split}' no soportado. "
                f"Opciones: {list(LIBRISPEECH_URLS.keys())}"
            )

        os.makedirs(self.root_dir, exist_ok=True)
        url = LIBRISPEECH_URLS[self.split]
        tar_path = os.path.join(self.root_dir, f"{self.split}.tar.gz")

        if not os.path.isfile(tar_path):
            print(f"Descargando {self.split} desde {url}...")
            print("Esto puede tardar varios minutos dependiendo de tu conexión.")
            urllib.request.urlretrieve(url, tar_path, reporthook=self._download_progress)
            print()  # Nueva línea después de la barra de progreso

        print(f"Extrayendo {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.root_dir)

        # Limpiar archivo tar
        os.remove(tar_path)
        print(f"Dataset extraído en '{dataset_dir}'")
        return dataset_dir

    @staticmethod
    def _download_progress(count, block_size, total_size):
        """Callback para mostrar progreso de descarga."""
        percent = count * block_size * 100 // total_size
        print(f"\rProgreso: {percent}%", end="", flush=True)

    def load_samples(self) -> list[tuple[str, str]]:
        """
        Carga los pares (audio_path, transcript) desde el dataset.

        Returns:
            Lista de tuplas (audio_path, transcript).
        """
        dataset_dir = os.path.join(self.root_dir, "LibriSpeech", self.split)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(
                f"Dataset no encontrado en '{dataset_dir}'. "
                "Ejecuta download() primero."
            )

        self.samples = []
        # Buscar todos los archivos de transcripción
        trans_files = glob.glob(
            os.path.join(dataset_dir, "**", "*.trans.txt"), recursive=True
        )

        for trans_file in sorted(trans_files):
            dir_path = os.path.dirname(trans_file)
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Formato: <utterance_id> <transcript>
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utterance_id, transcript = parts
                    audio_path = os.path.join(dir_path, f"{utterance_id}.flac")

                    if os.path.isfile(audio_path):
                        transcript = transcript.lower().strip()
                        # Filtrar por longitud de etiqueta
                        if self.max_label_len and len(transcript) > self.max_label_len:
                            continue
                        self.samples.append((audio_path, transcript))

                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break
            if self.max_samples and len(self.samples) >= self.max_samples:
                break

        print(f"Cargadas {len(self.samples)} muestras del split '{self.split}'")
        return self.samples

    def _process_sample(self, audio_path: str, transcript: str):
        """
        Procesa una muestra: audio → espectrograma, texto → enteros.

        Args:
            audio_path: Ruta al archivo de audio.
            transcript: Transcripción de texto.

        Returns:
            Tupla (spectrogram, label) o None si hay error.
        """
        try:
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
            # Normalizar
            mel_db = AudioProcessor.normalize(mel_db)
            # Transponer: (n_mels, time) -> (time, n_mels)
            mel_db = mel_db.T

            # Truncar si es necesario
            if self.max_audio_len and mel_db.shape[0] > self.max_audio_len:
                mel_db = mel_db[:self.max_audio_len, :]

            label = self.encoder.encode(transcript)
            return mel_db.astype(np.float32), np.array(label, dtype=np.int32)
        except Exception as e:
            print(f"Error procesando {audio_path}: {e}")
            return None

    def create_tf_dataset(self, batch_size: int = 16, shuffle: bool = True) -> tf.data.Dataset:
        """
        Crea un tf.data.Dataset con padding dinámico para entrenamiento.

        Usa procesamiento lazy: los audios se procesan uno a uno bajo demanda
        en lugar de precargar todo en memoria.

        El dataset devuelve tuplas con:
            - espectrograma mel (time, n_mels)
            - etiquetas codificadas (enteros)
            - input_length: longitud temporal del espectrograma
            - label_length: longitud de la etiqueta

        Args:
            batch_size: Tamaño de batch.
            shuffle: Si se mezclan los datos.

        Returns:
            tf.data.Dataset listo para entrenamiento.
        """
        if not self.samples:
            self.load_samples()

        # Filtrar muestras válidas con preprocesamiento rápido
        # Solo guardamos rutas y transcripciones (strings livianos)
        valid_samples = []
        print("Validando muestras de audio...")
        for i, (audio_path, transcript) in enumerate(self.samples):
            if os.path.isfile(audio_path) and len(transcript.strip()) > 0:
                valid_samples.append((audio_path, transcript))
            if (i + 1) % 2000 == 0:
                print(f"  Validadas {i + 1}/{len(self.samples)} muestras...")

        print(f"Muestras válidas: {len(valid_samples)}")

        # Capturar parámetros para el closure
        sample_rate = self.sample_rate
        n_mels = self.n_mels
        n_fft = self.processor.n_fft
        hop_length = self.processor.hop_length
        fmin = self.processor.fmin
        fmax = self.processor.fmax
        max_audio_len = self.max_audio_len
        encoder = self.encoder

        def _process_lazy(audio_path_tensor, transcript_tensor):
            """Procesa una muestra de forma lazy dentro de tf.py_function."""
            audio_path_str = audio_path_tensor.numpy().decode("utf-8")
            transcript_str = transcript_tensor.numpy().decode("utf-8")

            try:
                audio, sr = librosa.load(audio_path_str, sr=sample_rate)
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    fmin=fmin,
                    fmax=fmax,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = AudioProcessor.normalize(mel_db)
                mel_db = mel_db.T  # (n_mels, time) -> (time, n_mels)

                # Truncar si es necesario
                if max_audio_len and mel_db.shape[0] > max_audio_len:
                    mel_db = mel_db[:max_audio_len, :]

                label = encoder.encode(transcript_str)

                spec = mel_db.astype(np.float32)
                lab = np.array(label, dtype=np.int32)
                il = np.array([spec.shape[0]], dtype=np.int32)
                ll = np.array([len(label)], dtype=np.int32)

                return spec, lab, il, ll
            except Exception as e:
                # Devolver arrays vacíos en caso de error (se filtrarán)
                return (
                    np.zeros((1, n_mels), dtype=np.float32),
                    np.zeros((1,), dtype=np.int32),
                    np.array([0], dtype=np.int32),
                    np.array([0], dtype=np.int32),
                )

        def _tf_process(audio_path, transcript):
            """Wrapper de tf.py_function para procesamiento lazy."""
            spec, lab, il, ll = tf.py_function(
                _process_lazy,
                [audio_path, transcript],
                [tf.float32, tf.int32, tf.int32, tf.int32],
            )
            # Establecer shapes conocidas para que padded_batch funcione
            spec.set_shape([None, n_mels])
            lab.set_shape([None])
            il.set_shape([1])
            ll.set_shape([1])
            return spec, lab, il, ll

        # Crear dataset desde las rutas y transcripciones (livianas en memoria)
        audio_paths = [s[0] for s in valid_samples]
        transcripts = [s[1] for s in valid_samples]

        dataset = tf.data.Dataset.from_tensor_slices((audio_paths, transcripts))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(2000, len(valid_samples)))

        # Procesamiento lazy: cada muestra se procesa bajo demanda
        dataset = dataset.map(
            _tf_process,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Filtrar muestras con error (input_length == 0)
        dataset = dataset.filter(lambda s, l, il, ll: tf.greater(il[0], 0))

        # Determinar padded_shapes: usar max_audio_len si está definido
        max_spec_len = max_audio_len if max_audio_len else None
        max_lab_len = self.max_label_len if self.max_label_len else None

        # Padded batch: alinea las secuencias con padding
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [max_spec_len, n_mels],   # spectrograms (cap máximo)
                [max_lab_len],              # labels
                [1],                        # input_length
                [1],                        # label_length
            ),
            padding_values=(
                0.0,                                    # pad spectrograms con 0
                np.int32(encoder.BLANK_TOKEN),          # pad labels con blank
                np.int32(0),                            # input_length
                np.int32(0),                            # label_length
            ),
            drop_remainder=True,  # Evitar batches parciales con shapes diferentes
        )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
