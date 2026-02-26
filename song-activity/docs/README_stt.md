# Speech-to-Text (STT / ASR)

Reconocimiento automático de voz basado en **DeepSpeech 2**, implementado en PyTorch con soporte CUDA.

## Arquitectura

```
Audio WAV → Mel Spectrogram → CNN → Bi-LSTM → Dense → CTC Loss → Texto
```

| Capa | Función |
|------|---------|
| **Conv2D × 2** | Extrae features locales del espectrograma |
| **Bi-LSTM × 2** | Captura dependencias temporales bidireccionales |
| **Dense + LogSoftmax** | Proyecta al vocabulario + blank token |
| **CTC Loss** | Alinea audio y texto sin alineación manual |

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `model.py` | `ASRModel` (DeepSpeech 2 en PyTorch) |
| `trainer.py` | Loop de entrenamiento con CUDA y mixed precision |
| `predictor.py` | Inferencia con decodificación greedy/beam |
| `audio_processor.py` | Espectrogramas mel (librosa, 16kHz, 128 bandas) |
| `text_encoder.py` | Codificación texto ↔ enteros, decodificación CTC |
| `dataset.py` | Dataset LibriSpeech (PyTorch DataLoader) |

## Entrenamiento

```bash
python main.py train \
    --data_dir ./data \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --save_path ./model_saved
```

### Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--data_dir` | `./data` | Directorio para LibriSpeech |
| `--epochs` | 50 | Épocas de entrenamiento |
| `--batch_size` | 32 | Tamaño de batch |
| `--learning_rate` | 1e-3 | Tasa de aprendizaje |
| `--rnn_units` | 256 | Unidades por capa LSTM |
| `--max_samples` | 3000 | Límite de muestras |

### Output del Entrenamiento

```
model_saved/
├── final_model.pt              # Pesos finales
├── training_history.png        # Curva de loss
└── checkpoints/
    ├── best_model.pt           # Mejor modelo (val loss)
    └── epoch_10.pt             # Checkpoints periódicos
```

## Predicción / Transcripción

```bash
python main.py predict \
    --model_path ./model_saved/final_model.pt \
    --input cancion.mp4 \
    --output ./output/transcription.txt \
    --target_lang es
```

### Outputs Generados (7 archivos)

1. Transcripción en inglés (`.txt`)
2. Traducción al español (`.txt`)
3. Audio de vocales aisladas (`.wav`)
4. Datos mel spectrogram (`.npy`)
5. Imagen mel spectrogram (`.png`)
6. Fonética IPA (`.txt`)
7. Análisis de ritmo (`.json`)
