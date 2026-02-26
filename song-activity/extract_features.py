"""
Extract Features: Extracts pitch (F0), energy, and durations from audio segments.
Required for FastSpeech 2 training.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import librosa


def extract_pitch(
    wav_path: str,
    sr: int = 22050,
    hop_length: int = 256,
    fmin: float = 80.0,
    fmax: float = 800.0,
) -> np.ndarray:
    """
    Extracts pitch (F0) from audio using pYIN algorithm.

    Args:
        wav_path: Path to the audio file.
        sr: Sample rate.
        hop_length: Hop length for frame-level analysis.
        fmin: Minimum expected frequency.
        fmax: Maximum expected frequency.

    Returns:
        Pitch contour (T,) with NaN for unvoiced frames.
    """
    audio, _ = librosa.load(wav_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
    )
    # Replace NaN with 0 for unvoiced frames
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


def extract_energy(
    wav_path: str,
    sr: int = 22050,
    hop_length: int = 256,
    n_fft: int = 1024,
) -> np.ndarray:
    """
    Extracts frame-level energy (RMS) from audio.

    Args:
        wav_path: Path to the audio file.
        sr: Sample rate.
        hop_length: Hop length.
        n_fft: FFT window size.

    Returns:
        Energy contour (T,).
    """
    audio, _ = librosa.load(wav_path, sr=sr)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    energy = np.sqrt(np.sum(np.abs(stft) ** 2, axis=0))
    return energy.astype(np.float32)


def extract_durations_simple(
    wav_path: str,
    transcript: str,
    sr: int = 22050,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Estimates phoneme-level durations using uniform distribution.
    A simple baseline alignment (real alignment would use MFA or similar).

    Args:
        wav_path: Path to the audio file.
        transcript: Text transcription.
        sr: Sample rate.
        hop_length: Hop length.

    Returns:
        Duration array (num_chars,) in frames.
    """
    audio, _ = librosa.load(wav_path, sr=sr)
    total_frames = 1 + len(audio) // hop_length
    n_chars = len(transcript.strip())

    if n_chars == 0:
        return np.array([], dtype=np.int32)

    # Uniform distribution of frames across characters
    base_dur = total_frames // n_chars
    remainder = total_frames - base_dur * n_chars
    durations = np.full(n_chars, base_dur, dtype=np.int32)
    # Distribute remainder evenly
    for i in range(remainder):
        durations[i] += 1

    return durations


def extract_all_features(
    wav_path: str,
    output_dir: str,
    segment_name: str,
    transcript: str = None,
    sr: int = 22050,
    hop_length: int = 256,
) -> dict:
    """
    Extracts and saves all features (pitch, energy, durations) for a segment.

    Args:
        wav_path: Path to the audio segment.
        output_dir: Base output directory.
        segment_name: Segment identifier (e.g., 'segment_001').
        transcript: Text transcription (needed for durations).
        sr: Sample rate.
        hop_length: Hop length.

    Returns:
        Dict with paths to saved feature files.
    """
    pitch_dir = os.path.join(output_dir, "pitch")
    energy_dir = os.path.join(output_dir, "energy")
    duration_dir = os.path.join(output_dir, "durations")
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)
    os.makedirs(duration_dir, exist_ok=True)

    # Extract features
    pitch = extract_pitch(wav_path, sr=sr, hop_length=hop_length)
    energy = extract_energy(wav_path, sr=sr, hop_length=hop_length)

    # Save
    pitch_path = os.path.join(pitch_dir, f"{segment_name}.npy")
    energy_path = os.path.join(energy_dir, f"{segment_name}.npy")
    np.save(pitch_path, pitch)
    np.save(energy_path, energy)

    result = {"pitch": pitch_path, "energy": energy_path}

    if transcript:
        durations = extract_durations_simple(wav_path, transcript, sr=sr, hop_length=hop_length)
        duration_path = os.path.join(duration_dir, f"{segment_name}.npy")
        np.save(duration_path, durations)
        result["durations"] = duration_path

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pitch, energy, and duration features")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory with WAV segments")
    parser.add_argument("--transcript_file", type=str, default=None, help="Transcripts file (id|text)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    args = parser.parse_args()

    # Load transcripts if provided
    transcripts = {}
    if args.transcript_file and os.path.exists(args.transcript_file):
        with open(args.transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]

    wav_files = sorted([f for f in os.listdir(args.wav_dir) if f.endswith(".wav")])
    print(f"Extracting features for {len(wav_files)} files...")

    for wav_file in wav_files:
        name = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(args.wav_dir, wav_file)
        transcript = transcripts.get(name, None)

        result = extract_all_features(
            wav_path, args.output_dir, name, transcript=transcript,
        )
        print(f"  {name}: pitch={os.path.exists(result['pitch'])}, "
              f"energy={os.path.exists(result['energy'])}")

    print("Feature extraction complete!")
