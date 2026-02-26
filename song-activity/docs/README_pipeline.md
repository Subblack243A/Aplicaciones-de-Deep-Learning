# Pipeline Completo (End-to-End)

Ejecuta el pipeline completo: desde un archivo de audio/video hasta obtener una versión cantada sintetizada.

## Flujo

```
┌─────────────────────────────────────────────────────────────────┐
│  Audio/Video (.mp4, .wav, ...)                                  │
│       │                                                         │
│       ▼ [1] Conversión + Separación de vocales                  │
│  WAV vocals (16kHz, mono)                                       │
│       │                                                         │
│       ▼ [2] STT (DeepSpeech 2)                                  │
│  Transcripción (.txt) + Mel + Fonética + Ritmo                  │
│       │                                                         │
│       ▼ [3] Traducción (argostranslate)                         │
│  Texto traducido (.txt)                                         │
│       │                                                         │
│       ▼ [4] SVS (Tacotron 2 / FastSpeech 2)                    │
│  Mel Spectrogram sintético                                      │
│       │                                                         │
│       ▼ [5] Vocoder (Griffin-Lim / HiFi-GAN)                   │
│  Audio cantado (.wav)                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Uso

### Pipeline Completo

```bash
python main.py pipeline \
    --input cancion.mp4 \
    --asr_model ./model_saved/final_model.pt \
    --svs_model ./svs_saved/tacotron2/final_model.pt \
    --svs_type tacotron2 \
    --target_lang es \
    --output ./output/pipeline_output.wav
```

### Paso a Paso (Manual)

```bash
# 1. Transcribir
python main.py predict \
    --model_path ./model_saved/final_model.pt \
    --input cancion.mp4 \
    --output ./output/transcription.txt

# 2. Preparar dataset SVS
python prepare_dataset.py \
    --vocals_wav ./output/transcription_vocals.wav \
    --transcript ./output/transcription.txt \
    --output_dir ./dataset/

# 3. Entrenar SVS (si no hay modelo)
python main.py train-svs \
    --model tacotron2 \
    --dataset_dir ./dataset \
    --epochs 500

# 4. Sintetizar
python main.py synthesize \
    --model_type tacotron2 \
    --model_path ./svs_saved/tacotron2/final_model.pt \
    --input ./output/transcription.txt \
    --output ./output/singing.wav
```

## Parámetros del Pipeline

| Parámetro | Descripción |
|-----------|-------------|
| `--input` | Archivo de audio/video a procesar |
| `--asr_model` | Modelo ASR entrenado (.pt) |
| `--svs_model` | Modelo SVS entrenado (.pt, opcional) |
| `--svs_type` | `tacotron2` o `fastspeech2` |
| `--target_lang` | Idioma de traducción (`es`, `fr`, `none`) |
| `--output` | Ruta del audio de salida |

## Outputs Generados

```
output/
├── pipeline_output.wav           # Audio cantado (SVS)
├── pipeline_output_mel.png       # Mel spectrogram generado
├── pipeline_output_transcription.txt  # Transcripción STT
└── pipeline_output_translated.txt     # Traducción
```
