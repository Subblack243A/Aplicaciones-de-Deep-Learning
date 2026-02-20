"""
Main CLI: Punto de entrada para entrenar y predecir con el modelo ASR.

Uso:
  python main.py train --data_dir ./data --epochs 50 --batch_size 16 --save_path ./model_saved
  python main.py predict --model_path ./model_saved/best_model.keras --input cancion.mp4
"""

import argparse
import os
import sys
import tensorflow as tf

from text_encoder import TextEncoder
from audio_processor import AudioProcessor
from model import ASRModel
from dataset import LibriSpeechDataset
from trainer import Trainer
from predictor import Predictor

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

def train(args):
    """Ejecuta el pipeline de entrenamiento completo."""

    print("\n" + "=" * 60)
    print("  ASR Training Pipeline - DeepSpeech 2")
    print("=" * 60)

    # === Verificar GPU ===
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n  ✓ GPU detectada: {gpus}")
        # Permitir crecimiento de memoria para DirectML
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("\n  ✗ No se detectó GPU. Entrenando en CPU.")
        print("    Verifica: pip install tensorflow-cpu==2.10 tensorflow-directml-plugin")

    encoder = TextEncoder()
    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )

    # === 1. Cargar datasets ===
    print("\n[1/4] Cargando datasets...")

    train_data = LibriSpeechDataset(
        root_dir=args.data_dir,
        split=args.train_split,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        max_audio_len=args.max_audio_len,
        max_label_len=args.max_label_len,
        max_samples=args.max_samples,
    )
    train_data.download()
    train_data.load_samples()
    train_dataset, train_steps = train_data.create_tf_dataset(batch_size=args.batch_size)

    val_dataset = None
    if args.val_split:
        val_data = LibriSpeechDataset(
            root_dir=args.data_dir,
            split=args.val_split,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            max_audio_len=args.max_audio_len,
            max_label_len=args.max_label_len,
            max_samples=args.max_val_samples,
        )
        val_data.download()
        val_data.load_samples()
        val_dataset, _ = val_data.create_tf_dataset(
            batch_size=args.batch_size, shuffle=False
        )

    # === 2. Construir modelo ===
    print("\n[2/4] Construyendo modelo...")

    input_shape = (None, args.n_mels)  # (time_steps variable, n_mels)
    asr = ASRModel(rnn_units=args.rnn_units, dropout=args.dropout)
    model = asr.build(input_shape=input_shape, vocab_size=encoder.vocab_size)
    asr.summary()

    # === 3. Entrenar ===
    print("\n[3/4] Entrenando...")

    trainer = Trainer(
        model=model,
        encoder=encoder,
        learning_rate=args.learning_rate,
    )
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        checkpoint_dir=os.path.join(args.save_path, "checkpoints"),
        save_best=True,
        total_batches=train_steps,
    )

    # === 4. Guardar modelo y resultados ===
    print("\n[4/4] Guardando modelo y resultados...")

    final_model_path = os.path.join(args.save_path, "final_model.keras")
    trainer.save_model(model, final_model_path)

    # Guardar gráfica de entrenamiento
    plot_path = os.path.join(args.save_path, "training_history.png")
    trainer.plot_history(save_path=plot_path)

    print("\n" + "=" * 60)
    print("  Entrenamiento completado exitosamente!")
    print(f"  Modelo final: {final_model_path}")
    print(f"  Mejor modelo: {os.path.join(args.save_path, 'checkpoints', 'best_model.keras')}")
    print(f"  Gráfica: {plot_path}")
    print("=" * 60 + "\n")


def predict(args):
    """Ejecuta predicción sobre un archivo de audio/video."""
    print("\n" + "=" * 60)
    print("  ASR Prediction - DeepSpeech 2")
    print("=" * 60)

    encoder = TextEncoder()
    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )

    # Cargar modelo y crear predictor
    predictor_instance = Predictor(
        model_path=args.model_path,
        encoder=encoder,
        processor=processor,
    )

    # Transcribir
    print(f"\nTranscribiendo: '{args.input}'...")
    beam_width = args.beam_width if args.beam_width > 0 else None

    result = predictor_instance.transcribe(args.input, beam_width=beam_width)

    print(f"\n{'='*60}")
    print("  Transcripción:")
    print(f"{'='*60}")
    print(f"\n{result}\n")

    # Calcular WER si se proporciona referencia
    if args.reference:
        wer = Predictor.calculate_wer(args.reference, result)
        if wer is not None:
            print(f"WER (Word Error Rate): {wer:.4f} ({wer*100:.1f}%)\n")

    # Guardar resultado si se especifica
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Transcripción guardada en '{args.output}'")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ASR Song Transcription - DeepSpeech 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py train --data_dir ./data --epochs 50 --save_path ./model_saved
  python main.py predict --model_path ./model_saved/final_model.keras --input cancion.mp4
  python main.py predict --model_path ./model_saved/final_model.keras --input cancion.wav --beam_width 10
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # === Subcomando: train ===
    train_parser = subparsers.add_parser("train", help="Entrenar el modelo ASR")
    train_parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directorio para datos de LibriSpeech (default: ./data)"
    )
    train_parser.add_argument(
        "--train_split", type=str, default="train-clean-100",
        help="Split de entrenamiento (default: train-clean-100)"
    )
    train_parser.add_argument(
        "--val_split", type=str, default="dev-clean",
        help="Split de validación (default: dev-clean, None para deshabilitar)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50,
        help="Número de épocas (default: 50)"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Tamaño de batch (default: 32)"
    )
    train_parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Tasa de aprendizaje (default: 1e-3)"
    )
    train_parser.add_argument(
        "--rnn_units", type=int, default=256,
        help="Unidades LSTM (default: 256)"
    )
    train_parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout en LSTM (default: 0.1)"
    )
    train_parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Tasa de muestreo (default: 16000)"
    )
    train_parser.add_argument(
        "--n_mels", type=int, default=128,
        help="Bandas mel (default: 128)"
    )
    train_parser.add_argument(
        "--max_audio_len", type=int, default=1600,
        help="Longitud máxima de audio en time steps (default: 1600, ~16s)"
    )
    train_parser.add_argument(
        "--max_label_len", type=int, default=200,
        help="Longitud máxima de etiquetas en caracteres (default: 200)"
    )
    train_parser.add_argument(
        "--max_samples", type=int, default=3000,
        help="Número máximo de muestras de entrenamiento (default: 3000, None para todas)"
    )
    train_parser.add_argument(
        "--max_val_samples", type=int, default=None,
        help="Número máximo de muestras de validación (default: todas)"
    )
    train_parser.add_argument(
        "--save_path", type=str, default="./model_saved",
        help="Directorio para guardar el modelo (default: ./model_saved)"
    )

    # === Subcomando: predict ===
    predict_parser = subparsers.add_parser("predict", help="Transcribir audio/video")
    predict_parser.add_argument(
        "--model_path", type=str, required=True,
        help="Ruta al modelo guardado (.keras)"
    )
    predict_parser.add_argument(
        "--input", type=str, required=True,
        help="Archivo de audio/video a transcribir"
    )
    predict_parser.add_argument(
        "--beam_width", type=int, default=0,
        help="Ancho de beam search (0 = greedy, default: 0)"
    )
    predict_parser.add_argument(
        "--reference", type=str, default=None,
        help="Texto de referencia para calcular WER"
    )
    predict_parser.add_argument(
        "--output", type=str, default=None,
        help="Archivo para guardar la transcripción"
    )
    predict_parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Tasa de muestreo (default: 16000)"
    )
    predict_parser.add_argument(
        "--n_mels", type=int, default=128,
        help="Bandas mel (default: 128)"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
