"""
Prepare Dataset: Segments vocal audio and creates the dataset structure
required for SVS training from ASR pipeline outputs.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import librosa
import soundfile as sf

from svs_audio_processor import SVSAudioProcessor
from extract_features import extract_all_features


def segment_audio_by_silence(
    wav_path: str,
    sr: int = 22050,
    top_db: int = 30,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
) -> list[tuple[np.ndarray, float, float]]:
    """
    Segments audio by detecting silence regions.

    Args:
        wav_path: Path to the audio file.
        sr: Sample rate.
        top_db: Threshold for silence detection (lower = more aggressive).
        min_duration: Minimum segment duration in seconds.
        max_duration: Maximum segment duration in seconds.

    Returns:
        List of (audio_segment, start_time, end_time) tuples.
    """
    audio, _ = librosa.load(wav_path, sr=sr)
    intervals = librosa.effects.split(audio, top_db=top_db)

    segments = []
    for start_sample, end_sample in intervals:
        start_time = start_sample / sr
        end_time = end_sample / sr
        duration = end_time - start_time

        if min_duration <= duration <= max_duration:
            segment = audio[start_sample:end_sample]
            segments.append((segment, start_time, end_time))

    return segments


def segment_by_transcript_lines(
    wav_path: str,
    transcript: str,
    sr: int = 22050,
) -> list[tuple[np.ndarray, str]]:
    """
    Segments audio uniformly based on transcript lines.
    Each line gets an equal portion of the audio.

    Args:
        wav_path: Path to the audio file.
        transcript: Full transcript text (one line per segment).
        sr: Sample rate.

    Returns:
        List of (audio_segment, line_text) tuples.
    """
    audio, _ = librosa.load(wav_path, sr=sr)
    lines = [l.strip() for l in transcript.strip().splitlines() if l.strip()]

    if not lines:
        return []

    total_samples = len(audio)
    samples_per_line = total_samples // len(lines)

    segments = []
    for i, line in enumerate(lines):
        start = i * samples_per_line
        end = start + samples_per_line if i < len(lines) - 1 else total_samples
        segment = audio[start:end]
        segments.append((segment, line))

    return segments


def prepare_dataset(
    vocals_wav: str,
    transcript_path: str,
    output_dir: str,
    sr: int = 22050,
) -> None:
    """
    Creates the full SVS dataset structure from vocal audio and transcript.

    Args:
        vocals_wav: Path to the vocals-only WAV file.
        transcript_path: Path to the transcript text file.
        output_dir: Output directory for the dataset.
        sr: Target sample rate.
    """
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # Read transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Segment audio by transcript lines
    print(f"Segmenting audio from '{vocals_wav}'...")
    segments = segment_by_transcript_lines(vocals_wav, transcript, sr=sr)
    print(f"  Created {len(segments)} segments")

    # Create transcripts file and save segments
    transcripts_lines = []
    processor = SVSAudioProcessor(sample_rate=sr)

    for i, (audio_segment, line_text) in enumerate(segments):
        segment_name = f"segment_{i+1:03d}"
        wav_path = os.path.join(wavs_dir, f"{segment_name}.wav")

        # Save segment audio
        sf.write(wav_path, audio_segment, sr)
        transcripts_lines.append(f"{segment_name}|{line_text}")

        # Extract features for this segment
        extract_all_features(
            wav_path=wav_path,
            output_dir=output_dir,
            segment_name=segment_name,
            transcript=line_text,
            sr=sr,
        )

    # Save transcripts file
    transcripts_path = os.path.join(output_dir, "transcripts.txt")
    with open(transcripts_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcripts_lines))

    print(f"\nDataset created successfully at '{output_dir}':")
    print(f"  Segments:    {len(segments)}")
    print(f"  Transcripts: {transcripts_path}")
    print(f"  WAVs:        {wavs_dir}/")
    print(f"  Pitch:       {os.path.join(output_dir, 'pitch')}/")
    print(f"  Energy:      {os.path.join(output_dir, 'energy')}/")
    print(f"  Durations:   {os.path.join(output_dir, 'durations')}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SVS dataset from ASR outputs")
    parser.add_argument("--vocals_wav", type=str, required=True,
                        help="Path to vocals-only WAV file")
    parser.add_argument("--transcript", type=str, required=True,
                        help="Path to transcript text file")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                        help="Output directory for the dataset")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Target sample rate (default: 22050)")
    args = parser.parse_args()

    prepare_dataset(
        vocals_wav=args.vocals_wav,
        transcript_path=args.transcript,
        output_dir=args.output_dir,
        sr=args.sample_rate,
    )
