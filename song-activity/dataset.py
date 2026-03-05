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

    def create_tf_dataset(self, batch_size: int = 16, shuffle: bool = True) -> tuple[tf.data.Dataset, int]:
        """
        Crea un tf.data.Dataset con padding dinámico para entrenamiento.

        Pre-procesa todos los audios a numpy arrays en memoria y luego crea
        tensores puros de TensorFlow. Esto elimina tf.py_function y permite
        que todo el pipeline de datos se ejecute en GPU.

        Args:
            batch_size: Tamaño de batch.
            shuffle: Si se mezclan los datos.

        Returns:
            Tupla (dataset, num_batches).
        """
        if not self.samples:
            self.load_samples()

        # === Pre-procesar todas las muestras a numpy arrays ===
        print("Pre-procesando audio a espectrogramas (esto se hace una vez)...")
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        errors = 0

        for i, (audio_path, transcript) in enumerate(self.samples):
            result = self._process_sample(audio_path, transcript)
            if result is not None:
                spec, lab = result
                spectrograms.append(spec)
                labels.append(lab)
                input_lengths.append(spec.shape[0])
                label_lengths.append(len(lab))
            else:
                errors += 1

            if (i + 1) % 500 == 0:
                print(f"  Procesadas {i + 1}/{len(self.samples)} muestras...")

        print(f"Pre-procesamiento completado: {len(spectrograms)} válidas, {errors} errores")

        if len(spectrograms) == 0:
            raise ValueError("No se pudieron procesar muestras válidas.")

        # === Pad arrays a longitud máxima para crear tensores uniformes ===
        max_spec_len = self.max_audio_len if self.max_audio_len else max(s.shape[0] for s in spectrograms)
        max_lab_len = self.max_label_len if self.max_label_len else max(len(l) for l in labels)

        # Crear arrays con padding
        n_samples = len(spectrograms)
        padded_specs = np.zeros((n_samples, max_spec_len, self.n_mels), dtype=np.float32)
        padded_labels = np.full((n_samples, max_lab_len), self.encoder.BLANK_TOKEN, dtype=np.int32)
        il_array = np.zeros((n_samples, 1), dtype=np.int32)
        ll_array = np.zeros((n_samples, 1), dtype=np.int32)

        for i in range(n_samples):
            spec_len = min(spectrograms[i].shape[0], max_spec_len)
            lab_len = min(len(labels[i]), max_lab_len)
            padded_specs[i, :spec_len, :] = spectrograms[i][:spec_len, :]
            padded_labels[i, :lab_len] = labels[i][:lab_len]
            il_array[i, 0] = spec_len
            ll_array[i, 0] = lab_len

        # Liberar memoria de listas originales
        del spectrograms, labels, input_lengths, label_lengths

        print(f"Dataset shapes: specs={padded_specs.shape}, labels={padded_labels.shape}")
        print(f"  Memoria estimada: {padded_specs.nbytes / 1e9:.2f} GB")

        # === Crear tf.data.Dataset desde tensores (sin tf.py_function) ===
        dataset = tf.data.Dataset.from_tensor_slices((
            padded_specs,
            padded_labels,
            il_array,
            ll_array,
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(2000, n_samples))

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        num_batches = n_samples // batch_size
        return dataset, num_batches
