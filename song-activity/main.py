"""
Main CLI: Unified entry point for the complete ASR + SVS pipeline.

Subcommands:
  train       - Train ASR model (DeepSpeech 2)
  predict     - Transcribe audio/video with trained ASR
  train-svs   - Train SVS model (Tacotron 2 or FastSpeech 2)
  synthesize  - Generate singing voice from text
  pipeline    - Full end-to-end: audio → STT → translation → SVS → WAV
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"


# ═══════════════════════════════════════════════════════════════
#   UTILS
# ═══════════════════════════════════════════════════════════════

def print_device_info():
    """Prints CUDA/GPU information."""
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("  ✗ CUDA not available. Running on CPU.")


def generate_phonetics(text: str, output_path: str) -> str:
    """Generates IPA phonetic transcription line by line."""
    import eng_to_ipa

    lines = text.strip().splitlines()
    ipa_lines = [eng_to_ipa.convert(line) for line in lines]
    ipa_text = "\n".join(ipa_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ipa_text)
    print(f"IPA phonetics saved to '{output_path}'")
    return ipa_text


def generate_rhythm_analysis(audio: np.ndarray, sr: int, output_path: str) -> dict:
    """Generates rhythm analysis (BPM, beats) and saves as JSON."""
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    duration = len(audio) / sr

    tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

    rhythm = {
        "tempo_bpm": round(tempo_val, 2),
        "num_beats": int(len(beat_times)),
        "beat_times_seconds": [round(float(t), 3) for t in beat_times],
        "duration_seconds": round(duration, 2),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rhythm, f, indent=2, ensure_ascii=False)
    print(f"Rhythm analysis saved to '{output_path}'")
    return rhythm


# ═══════════════════════════════════════════════════════════════
#   COMMAND: train (ASR)
# ═══════════════════════════════════════════════════════════════

def cmd_train(args):
    """Executes the ASR training pipeline."""
    from text_encoder import TextEncoder
    from audio_processor import AudioProcessor
    from model import ASRModel
    from dataset import LibriSpeechDataset
    from trainer import Trainer

    print("\n" + "=" * 60)
    print("  ASR Training Pipeline - DeepSpeech 2 (PyTorch)")
    print("=" * 60)
    print_device_info()

    encoder = TextEncoder()

    # 1. Load datasets
    print("\n[1/4] Loading datasets...")
    train_data = LibriSpeechDataset(
        root_dir=args.data_dir, split=args.train_split,
        sample_rate=args.sample_rate, n_mels=args.n_mels,
        max_audio_len=args.max_audio_len, max_label_len=args.max_label_len,
        max_samples=args.max_samples,
    )
    train_data.download()
    train_data.load_samples()
    train_loader = train_data.create_dataloader(batch_size=args.batch_size)

    val_loader = None
    if args.val_split:
        val_data = LibriSpeechDataset(
            root_dir=args.data_dir, split=args.val_split,
            sample_rate=args.sample_rate, n_mels=args.n_mels,
            max_audio_len=args.max_audio_len, max_label_len=args.max_label_len,
            max_samples=args.max_val_samples,
        )
        val_data.download()
        val_data.load_samples()
        val_loader = val_data.create_dataloader(batch_size=args.batch_size, shuffle=False)

    # 2. Build model
    print("\n[2/4] Building model...")
    model = ASRModel(
        n_mels=args.n_mels, vocab_size=encoder.vocab_size,
        rnn_units=args.rnn_units, dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # 3. Train
    print("\n[3/4] Training...")
    trainer = Trainer(model=model, encoder=encoder, learning_rate=args.learning_rate)
    trainer.train(
        train_dataloader=train_loader, val_dataloader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=os.path.join(args.save_path, "checkpoints"),
    )

    # 4. Save
    print("\n[4/4] Saving model and results...")
    final_path = os.path.join(args.save_path, "final_model.pt")
    trainer.save_model(model, final_path)
    plot_path = os.path.join(args.save_path, "training_history.png")
    trainer.plot_history(save_path=plot_path)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Final model:  {final_path}")
    print(f"  History plot: {plot_path}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
#   COMMAND: predict (ASR)
# ═══════════════════════════════════════════════════════════════

def cmd_predict(args):
    """Executes prediction on audio/video and generates outputs."""
    from text_encoder import TextEncoder
    from audio_processor import AudioProcessor
    from predictor import Predictor
    from traduction import TranslationService

    print("\n" + "=" * 60)
    print("  ASR Prediction - DeepSpeech 2 (PyTorch)")
    print("=" * 60)
    print_device_info()

    encoder = TextEncoder()
    processor = AudioProcessor(sample_rate=args.sample_rate, n_mels=args.n_mels)

    predictor_instance = Predictor(
        model_path=args.model_path, encoder=encoder, processor=processor,
        n_mels=args.n_mels, rnn_units=args.rnn_units,
    )

    print(f"\nTranscribing: '{args.input}'...")
    beam_width = args.beam_width if args.beam_width > 0 else None
    artifacts = predictor_instance.transcribe_full(args.input, beam_width=beam_width)
    result = artifacts["transcription"]

    print(f"\n{'='*60}")
    print("  Transcription:")
    print(f"{'='*60}")
    print(f"\n{result}\n")

    if args.reference:
        wer = Predictor.calculate_wer(args.reference, result)
        if wer is not None:
            print(f"WER: {wer:.4f} ({wer*100:.1f}%)\n")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        base_name, _ = os.path.splitext(args.output)

        # 1. Transcription
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Transcription saved to '{args.output}'")

        # 2. Translation
        if args.target_lang and args.target_lang.lower() != "none":
            print(f"\nTranslating to '{args.target_lang}'...")
            translator = TranslationService(from_code="en", to_code=args.target_lang)
            output_translated = f"{base_name}_{args.target_lang}.txt"
            translator.translate_file(args.output, output_translated, batch_size=5)
            if os.path.exists(output_translated):
                with open(output_translated, "r", encoding="utf-8") as f:
                    translated_text = f.read()
                print(f"\n{'='*60}")
                print(f"  Translation ({args.target_lang.upper()}):")
                print(f"{'='*60}\n{translated_text}\n")

        # 3. Vocals WAV
        vocals_output = f"{base_name}_vocals.wav"
        if artifacts["vocals_path"] and os.path.isfile(artifacts["vocals_path"]):
            shutil.copy2(artifacts["vocals_path"], vocals_output)

        # 4-5. Mel spectrogram
        np.save(f"{base_name}_mel_spectrogram.npy", artifacts["mel_spectrogram"])
        processor.save_spectrogram(artifacts["mel_spectrogram"], f"{base_name}_mel_spectrogram.png")

        # 6. Phonetics
        generate_phonetics(result, f"{base_name}_phonetics.txt")

        # 7. Rhythm
        generate_rhythm_analysis(artifacts["audio"], artifacts["sample_rate"], f"{base_name}_rhythm.json")

        print(f"\n{'='*60}")
        print("  All outputs generated successfully.")
        print(f"{'='*60}\n")

    return result


# ═══════════════════════════════════════════════════════════════
#   COMMAND: train-svs
# ═══════════════════════════════════════════════════════════════

def cmd_train_svs(args):
    """Trains a Singing Voice Synthesis model."""
    from svs_dataset import create_tacotron_dataloader, create_fastspeech_dataloader
    from svs_trainer import SVSTrainer

    print("\n" + "=" * 60)
    print(f"  SVS Training - {args.model.upper()} (PyTorch)")
    print("=" * 60)
    print_device_info()

    output_dir = os.path.join(args.save_path, args.model)

    if args.model == "tacotron2":
        from tacotron2 import Tacotron2, Tacotron2Config, Tacotron2Loss
        from svs_text_processor import SVSTextProcessor

        config = Tacotron2Config(vocab_size=SVSTextProcessor().vocab_size)
        model = Tacotron2(config)
        loss_fn = Tacotron2Loss()

        train_loader = create_tacotron_dataloader(args.dataset_dir, args.batch_size)
        val_loader = None

        trainer = SVSTrainer(model, loss_fn, args.learning_rate, "Tacotron 2", output_dir)
        trainer.train_tacotron2(train_loader, val_loader, args.epochs)

    elif args.model == "fastspeech2":
        from fastspeech2 import FastSpeech2, FastSpeech2Config, FastSpeech2Loss
        from svs_text_processor import SVSTextProcessor

        config = FastSpeech2Config(vocab_size=SVSTextProcessor().vocab_size)
        model = FastSpeech2(config)
        loss_fn = FastSpeech2Loss()

        train_loader = create_fastspeech_dataloader(args.dataset_dir, args.batch_size)
        val_loader = None

        trainer = SVSTrainer(model, loss_fn, args.learning_rate, "FastSpeech 2", output_dir)
        trainer.train_fastspeech2(train_loader, val_loader, args.epochs)

    else:
        print(f"Unknown model: {args.model}. Use 'tacotron2' or 'fastspeech2'.")
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")
    print(f"  Training outputs: {output_dir}/")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
#   COMMAND: synthesize
# ═══════════════════════════════════════════════════════════════

def cmd_synthesize(args):
    """Generates singing voice from text using a trained SVS model."""
    from svs_text_processor import SVSTextProcessor
    from svs_audio_processor import SVSAudioProcessor

    print("\n" + "=" * 60)
    print(f"  SVS Synthesis - {args.model_type.upper()}")
    print("=" * 60)
    print_device_info()

    text_proc = SVSTextProcessor()
    audio_proc = SVSAudioProcessor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read input text
    if os.path.isfile(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = args.input

    print(f"  Input text: {text[:80]}{'...' if len(text) > 80 else ''}")

    # Load model
    if args.model_type == "tacotron2":
        from tacotron2 import Tacotron2, Tacotron2Config

        config = Tacotron2Config(vocab_size=text_proc.vocab_size)
        model = Tacotron2(config).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Tokenize and synthesize
        tokens = text_proc.text_to_sequence(text)
        text_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        text_lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)

        result = model.inference(text_tensor, text_lengths)
        mel = result["mel_spectrogram"].squeeze(0).cpu().numpy()

    elif args.model_type == "fastspeech2":
        from fastspeech2 import FastSpeech2, FastSpeech2Config

        config = FastSpeech2Config(vocab_size=text_proc.vocab_size)
        model = FastSpeech2(config).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        tokens = text_proc.text_to_sequence(text)
        text_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        text_lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)

        result = model.inference(text_tensor, text_lengths)
        mel = result["mel_spectrogram"].squeeze(0).cpu().numpy()

    else:
        print(f"Unknown model type: {args.model_type}")
        sys.exit(1)

    # Convert mel to audio
    print("  Converting mel spectrogram to audio (Griffin-Lim)...")
    audio = audio_proc.mel_to_audio(mel)
    audio_proc.save_wav(audio, args.output)

    # Save mel visualization
    mel_png_path = os.path.splitext(args.output)[0] + "_mel.png"
    audio_proc.save_mel_plot(mel, mel_png_path)

    print(f"\n  Output audio: {args.output}")
    print(f"  Mel plot:     {mel_png_path}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
#   COMMAND: pipeline (Full end-to-end)
# ═══════════════════════════════════════════════════════════════

def cmd_pipeline(args):
    """Full pipeline: audio → STT → translation → SVS → singing WAV."""
    from predictor import Predictor
    from text_encoder import TextEncoder
    from audio_processor import AudioProcessor
    from traduction import TranslationService

    print("\n" + "=" * 60)
    print("  Full Pipeline: Audio → STT → Translation → SVS")
    print("=" * 60)
    print_device_info()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    base_name, _ = os.path.splitext(args.output)

    # Step 1: STT
    print("\n[1/3] Speech-to-Text...")
    encoder = TextEncoder()
    processor = AudioProcessor(sample_rate=16000, n_mels=128)
    predictor = Predictor(
        model_path=args.asr_model, encoder=encoder, processor=processor,
    )
    artifacts = predictor.transcribe_full(args.input)
    transcription = artifacts["transcription"]

    txt_path = f"{base_name}_transcription.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"  Transcription: {transcription[:100]}...")

    # Step 2: Translation
    translated_path = None
    if args.target_lang and args.target_lang.lower() != "none":
        print(f"\n[2/3] Translating to '{args.target_lang}'...")
        translator = TranslationService(from_code="en", to_code=args.target_lang)
        translated_path = f"{base_name}_translated.txt"
        translator.translate_file(txt_path, translated_path, batch_size=5)
    else:
        print("\n[2/3] Translation skipped.")

    # Step 3: SVS Synthesis
    if args.svs_model:
        print(f"\n[3/3] Synthesizing with {args.svs_type}...")
        # Build synthesize args
        synth_args = argparse.Namespace(
            model_type=args.svs_type,
            model_path=args.svs_model,
            input=txt_path,
            output=args.output,
        )
        cmd_synthesize(synth_args)
    else:
        print("\n[3/3] SVS skipped (no --svs_model provided).")

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Transcription: {txt_path}")
    if translated_path:
        print(f"  Translation:   {translated_path}")
    if args.svs_model:
        print(f"  Singing WAV:   {args.output}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
#   CLI PARSER
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Song Activity: ASR + SVS Pipeline (PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  train       Train ASR model (DeepSpeech 2)
  predict     Transcribe audio/video
  train-svs   Train SVS model (Tacotron 2 / FastSpeech 2)
  synthesize  Generate singing voice from text
  pipeline    Full end-to-end: audio → STT → translation → SVS
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ── train ──
    tp = subparsers.add_parser("train", help="Train ASR model")
    tp.add_argument("--data_dir", type=str, default="./data")
    tp.add_argument("--train_split", type=str, default="train-clean-100")
    tp.add_argument("--val_split", type=str, default="dev-clean")
    tp.add_argument("--epochs", type=int, default=50)
    tp.add_argument("--batch_size", type=int, default=32)
    tp.add_argument("--learning_rate", type=float, default=1e-3)
    tp.add_argument("--rnn_units", type=int, default=256)
    tp.add_argument("--dropout", type=float, default=0.1)
    tp.add_argument("--sample_rate", type=int, default=16000)
    tp.add_argument("--n_mels", type=int, default=128)
    tp.add_argument("--max_audio_len", type=int, default=1600)
    tp.add_argument("--max_label_len", type=int, default=200)
    tp.add_argument("--max_samples", type=int, default=3000)
    tp.add_argument("--max_val_samples", type=int, default=None)
    tp.add_argument("--save_path", type=str, default="./model_saved")

    # ── predict ──
    pp = subparsers.add_parser("predict", help="Transcribe audio/video")
    pp.add_argument("--model_path", type=str, required=True)
    pp.add_argument("--input", type=str, required=True)
    pp.add_argument("--beam_width", type=int, default=0)
    pp.add_argument("--reference", type=str, default=None)
    pp.add_argument("--output", type=str, default=None)
    pp.add_argument("--target_lang", type=str, default="es")
    pp.add_argument("--sample_rate", type=int, default=16000)
    pp.add_argument("--n_mels", type=int, default=128)
    pp.add_argument("--rnn_units", type=int, default=256)

    # ── train-svs ──
    sp = subparsers.add_parser("train-svs", help="Train SVS model")
    sp.add_argument("--model", type=str, required=True, choices=["tacotron2", "fastspeech2"])
    sp.add_argument("--dataset_dir", type=str, required=True)
    sp.add_argument("--epochs", type=int, default=500)
    sp.add_argument("--batch_size", type=int, default=16)
    sp.add_argument("--learning_rate", type=float, default=1e-4)
    sp.add_argument("--save_path", type=str, default="./svs_saved")

    # ── synthesize ──
    syp = subparsers.add_parser("synthesize", help="Generate singing from text")
    syp.add_argument("--model_type", type=str, required=True, choices=["tacotron2", "fastspeech2"])
    syp.add_argument("--model_path", type=str, required=True)
    syp.add_argument("--input", type=str, required=True, help="Text file or direct text")
    syp.add_argument("--output", type=str, default="./output/singing.wav")

    # ── pipeline ──
    pip = subparsers.add_parser("pipeline", help="Full end-to-end pipeline")
    pip.add_argument("--input", type=str, required=True, help="Input audio/video file")
    pip.add_argument("--asr_model", type=str, required=True, help="Path to trained ASR model (.pt)")
    pip.add_argument("--svs_model", type=str, default=None, help="Path to trained SVS model (.pt)")
    pip.add_argument("--svs_type", type=str, default="tacotron2", choices=["tacotron2", "fastspeech2"])
    pip.add_argument("--target_lang", type=str, default="es")
    pip.add_argument("--output", type=str, default="./output/pipeline_output.wav")

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "train-svs": cmd_train_svs,
        "synthesize": cmd_synthesize,
        "pipeline": cmd_pipeline,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
