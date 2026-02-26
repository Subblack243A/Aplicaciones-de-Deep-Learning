"""
Download TTS Dataset: Downloads and prepares LJSpeech for Tacotron 2 and FastSpeech 2 training.
LJSpeech: ~13,100 audio clips (22050Hz) with English text transcriptions from a single speaker.
"""

from __future__ import annotations
import os
import tarfile
import urllib.request
import argparse

from extract_features import extract_all_features

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def download_ljspeech(output_dir: str = "./data") -> str:
    """
    Downloads and extracts the LJSpeech dataset.

    Args:
        output_dir: Root directory for the dataset.

    Returns:
        Path to the extracted dataset.
    """
    dataset_path = os.path.join(output_dir, "LJSpeech-1.1")

    if os.path.exists(dataset_path):
        print(f"  ✓ LJSpeech already exists at '{dataset_path}'")
        return dataset_path

    os.makedirs(output_dir, exist_ok=True)
    tar_path = os.path.join(output_dir, "LJSpeech-1.1.tar.bz2")

    print("  Downloading LJSpeech-1.1 (~2.6 GB)...")
    urllib.request.urlretrieve(LJSPEECH_URL, tar_path, reporthook=_progress)
    print()

    print("  Extracting...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(output_dir)

    if os.path.isfile(tar_path):
        os.remove(tar_path)

    print(f"  ✓ LJSpeech extracted to '{dataset_path}'")
    return dataset_path


def _progress(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size)
    print(f"\r  Progress: {pct}%", end="", flush=True)


def prepare_ljspeech_for_svs(
    ljspeech_dir: str,
    output_dir: str = "./dataset",
    max_samples: int = None,
    extract_feats: bool = True,
) -> None:
    """
    Converts LJSpeech to the SVS training format:
      - wavs/ directory with audio segments
      - transcripts.txt with segment_name|text format
      - pitch/, energy/, durations/ feature directories

    Args:
        ljspeech_dir: Path to extracted LJSpeech-1.1 directory.
        output_dir: Output dataset directory.
        max_samples: Limit number of samples (None = use all ~13,100).
        extract_feats: Whether to extract pitch/energy/duration features.
    """
    wavs_src = os.path.join(ljspeech_dir, "wavs")
    metadata_path = os.path.join(ljspeech_dir, "metadata.csv")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at '{ljspeech_dir}'")

    wavs_dst = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dst, exist_ok=True)

    # Read metadata: LJSpeech format is "id|raw_text|normalized_text"
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                utt_id = parts[0]
                text = parts[2]  # Use normalized text
                wav_path = os.path.join(wavs_src, f"{utt_id}.wav")
                if os.path.isfile(wav_path):
                    samples.append((utt_id, text, wav_path))

    if max_samples:
        samples = samples[:max_samples]

    print(f"  Processing {len(samples)} samples...")

    # Create transcripts.txt and symlink/copy wavs
    transcripts_lines = []
    for i, (utt_id, text, wav_src_path) in enumerate(samples):
        wav_dst_path = os.path.join(wavs_dst, f"{utt_id}.wav")

        # Symlink instead of copy to save disk space
        if not os.path.exists(wav_dst_path):
            os.symlink(os.path.abspath(wav_src_path), wav_dst_path)

        transcripts_lines.append(f"{utt_id}|{text}")

        # Extract features for FastSpeech 2
        if extract_feats:
            extract_all_features(
                wav_path=wav_src_path,
                output_dir=output_dir,
                segment_name=utt_id,
                transcript=text,
            )

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(samples)} processed...")

    # Save transcripts
    transcripts_path = os.path.join(output_dir, "transcripts.txt")
    with open(transcripts_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcripts_lines))

    print(f"\n  ✓ Dataset ready at '{output_dir}':")
    print(f"    Samples:     {len(samples)}")
    print(f"    Transcripts: {transcripts_path}")
    print(f"    WAVs:        {wavs_dst}/")
    if extract_feats:
        print(f"    Pitch:       {os.path.join(output_dir, 'pitch')}/")
        print(f"    Energy:      {os.path.join(output_dir, 'energy')}/")
        print(f"    Durations:   {os.path.join(output_dir, 'durations')}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare LJSpeech for TTS training")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to download LJSpeech into")
    parser.add_argument("--dataset_dir", type=str, default="./dataset",
                        help="Directory for prepared SVS dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples (default: all ~13,100)")
    parser.add_argument("--skip_features", action="store_true",
                        help="Skip pitch/energy/duration extraction (faster, Tacotron2 only)")
    args = parser.parse_args()

    # Step 1: Download
    lj_path = download_ljspeech(args.output_dir)

    # Step 2: Prepare
    prepare_ljspeech_for_svs(
        ljspeech_dir=lj_path,
        output_dir=args.dataset_dir,
        max_samples=args.max_samples,
        extract_feats=not args.skip_features,
    )
