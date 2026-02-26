# Song Activity: ASR + Singing Voice Synthesis Pipeline

Pipeline completo de Deep Learning para transcripción automática de voz (ASR) y síntesis de voz cantada (SVS). Desde un archivo de audio/video, genera una versión cantada por una red neuronal.

## Overview del Pipeline

```
Audio/Video (MP4, WAV, ...)
     │
     ▼ [1] Conversión de audio
 WAV (16kHz mono)
     │
     ▼ [2] Separación de vocales
 Vocals WAV
     │
     ▼ [3] Speech-to-Text (DeepSpeech 2)
 Transcripción + Mel + Fonética + Ritmo
     │
     ▼ [4] Traducción (argostranslate)
 Texto traducido
     │
     ▼ [5] Singing Voice Synthesis (Tacotron 2 / FastSpeech 2)
 Mel Spectrogram sintético
     │
     ▼ [6] Vocoder (Griffin-Lim / HiFi-GAN)
 Audio cantado (.wav)
```

## Quick Start

### 1. Instalación

```bash
pip install -r requirements.txt
```

### 2. Entrenar ASR

```bash
python main.py train \
    --data_dir ./data \
    --epochs 50 \
    --batch_size 16 \
    --save_path ./model_saved
```

### 3. Transcribir Audio

```bash
python main.py predict \
    --model_path ./model_saved/final_model.pt \
    --input cancion.mp4 \
    --output ./output/transcription.txt
```

### 4. Entrenar SVS

```bash
python main.py train-svs \
    --model tacotron2 \
    --dataset_dir ./dataset \
    --epochs 500
```

### 5. Sintetizar Canto

```bash
python main.py synthesize \
    --model_type tacotron2 \
    --model_path ./svs_saved/tacotron2/final_model.pt \
    --input ./output/transcription.txt \
    --output ./output/singing.wav
```

### 6. Pipeline Completo

```bash
python main.py pipeline \
    --input cancion.mp4 \
    --asr_model ./model_saved/final_model.pt \
    --svs_model ./svs_saved/tacotron2/final_model.pt \
    --target_lang es \
    --output ./output/result.wav
```

## Estructura del Proyecto

```
song-activity/
├── main.py                  # CLI unificado (5 subcomandos)
│
├── ── ASR (Speech-to-Text) ──
├── model.py                 # DeepSpeech 2 (PyTorch)
├── trainer.py               # Training loop con CUDA + mixed precision
├── predictor.py             # Inferencia + transcripción
├── text_encoder.py          # Codificación CTC
├── dataset.py               # LibriSpeech dataset
├── audio_processor.py       # Mel spectrograms (16kHz, 128 bandas)
├── audio_converter.py       # Conversión MP4→WAV
├── audio_preprocessor.py    # Separación de vocales
│
├── ── SVS (Singing Voice Synthesis) ──
├── tacotron2.py             # Tacotron 2 completo
├── fastspeech2.py           # FastSpeech 2 completo
├── hifigan.py               # HiFi-GAN vocoder
├── svs_text_processor.py    # Tokenización para SVS
├── svs_audio_processor.py   # Audio SVS (22050Hz, 80 mels)
├── svs_dataset.py           # Datasets PyTorch
├── svs_trainer.py           # Training loops SVS
├── svs_visualizer.py        # Visualización del entrenamiento
├── extract_features.py      # Pitch, energy, duraciones
├── prepare_dataset.py       # Preparación de dataset
│
├── ── Utilidades ──
├── traduction.py            # Traducción local
├── enhance.py               # Speech enhancement
├── audio_utils_manual.py    # Utilidades de audio manuales
│
├── ── Documentación ──
├── docs/
│   ├── README_audio_conversion.md
│   ├── README_stt.md
│   ├── README_translation.md
│   ├── README_svs.md
│   └── README_pipeline.md
│
├── requirements.txt
└── README.md
```

## Documentación por Fase

| Fase | Documentación | Descripción |
|------|--------------|-------------|
| ⚡ Setup | [README_setup_training.md](docs/README_setup_training.md) | **Instalación + Entrenamiento GPU** |
| 1. Audio | [README_audio_conversion.md](docs/README_audio_conversion.md) | MP4 → WAV |
| 2. STT | [README_stt.md](docs/README_stt.md) | DeepSpeech 2 ASR |
| 3. Traducción | [README_translation.md](docs/README_translation.md) | Argostranslate |
| 4. SVS | [README_svs.md](docs/README_svs.md) | Tacotron 2 + FastSpeech 2 |
| 5. Pipeline | [README_pipeline.md](docs/README_pipeline.md) | End-to-end |

## Comandos CLI

```
python main.py --help

Subcomandos:
  train       Entrenar modelo ASR (DeepSpeech 2)
  predict     Transcribir audio/video
  train-svs   Entrenar modelo SVS (Tacotron 2 / FastSpeech 2)
  synthesize  Generar voz cantada desde texto
  pipeline    Pipeline completo: audio → STT → traducción → SVS
```

## Requisitos

- Python 3.10+
- PyTorch (con CUDA para GPU NVIDIA)
- FFMPEG (para conversión de audio/video)
- ~4GB VRAM mínimo para entrenamiento

## Framework

Todo el proyecto utiliza **PyTorch** como framework de Deep Learning unificado, con soporte nativo para:

- **CUDA**: Aceleración GPU NVIDIA
- **Mixed Precision**: `torch.amp` para reducir uso de VRAM
- **DataLoader**: Carga de datos paralela y eficiente
