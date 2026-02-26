# Singing Voice Synthesis (SVS)

Síntesis de voz cantada usando dos arquitecturas de redes neuronales: **Tacotron 2** (autoregresivo) y **FastSpeech 2** (no autoregresivo), implementados en PyTorch con soporte CUDA.

## Arquitecturas

### Tacotron 2 (Seq2Seq Autoregresivo)

```
Texto → Tokenización → Encoder (Embedding → Conv1D → BiLSTM)
    → Location-Sensitive Attention
    → Decoder (Autoregresivo, frame por frame)
    → PostNet (Conv1D residual)
    → Mel Spectrogram → Griffin-Lim → Audio WAV
```

**Loss**: MSE(mel_before_postnet) + MSE(mel_after_postnet) + BCE(gate)

### FastSpeech 2 (Transformer No Autoregresivo)

```
Texto → Tokenización → Encoder (FFT Blocks × 4)
    → Variance Adaptor:
        ├── Duration Predictor → Length Regulator
        ├── Pitch Predictor → Embedding
        └── Energy Predictor → Embedding
    → Decoder (FFT Blocks × 4)
    → Mel Linear → Mel Spectrogram → Griffin-Lim → Audio WAV
```

**Loss**: MSE(mel) + MSE(log_duration) + MSE(pitch) + MSE(energy)

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `tacotron2.py` | Encoder, Attention, Decoder, PostNet, Loss |
| `fastspeech2.py` | FFT Blocks, VarianceAdaptor, LengthRegulator, Loss |
| `hifigan.py` | Vocoder neuronal (Generator + Discriminators) |
| `svs_text_processor.py` | Tokenización fonética para SVS |
| `svs_audio_processor.py` | Audio SVS (22050Hz, 80 mels, Griffin-Lim) |
| `svs_dataset.py` | Datasets PyTorch (TacotronDataset, FastSpeechDataset) |
| `svs_trainer.py` | Loops de entrenamiento con visualización |
| `svs_visualizer.py` | Curvas de loss, comparación de espectrogramas |
| `extract_features.py` | Extracción de pitch, energy, duraciones |
| `prepare_dataset.py` | Preparación del dataset desde outputs ASR |

## Preparación del Dataset

```bash
python prepare_dataset.py \
    --vocals_wav ./output/snuff_transcription_vocals.wav \
    --transcript ./output/snuff_transcription.txt \
    --output_dir ./dataset/
```

Esto crea:

```
dataset/
├── wavs/                    # Segmentos de audio (22050 Hz)
├── transcripts.txt          # "segment_001|texto de la línea"
├── pitch/                   # F0 por frame (.npy)
├── energy/                  # Energía RMS por frame (.npy)
└── durations/               # Duraciones por fonema (.npy)
```

## Entrenamiento

### Tacotron 2

```bash
python main.py train-svs \
    --model tacotron2 \
    --dataset_dir ./dataset \
    --epochs 500 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --save_path ./svs_saved
```

### FastSpeech 2

```bash
python main.py train-svs \
    --model fastspeech2 \
    --dataset_dir ./dataset \
    --epochs 600 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --save_path ./svs_saved
```

### Output del Entrenamiento

```
svs_saved/tacotron2/
├── final_model.pt           # Pesos finales
├── best_model.pt            # Mejor modelo
├── learning_curves.png      # Gráficas de loss
├── training_report.md       # Reporte de entrenamiento
└── metrics.json             # Métricas programáticas
```

## Síntesis (Inferencia)

```bash
python main.py synthesize \
    --model_type tacotron2 \
    --model_path ./svs_saved/tacotron2/final_model.pt \
    --input ./output/snuff_transcription.txt \
    --output ./output/snuff_singing.wav
```

## Comparación

| Aspecto | Tacotron 2 | FastSpeech 2 |
|---------|-----------|-------------|
| **Datos requeridos** | Texto + audio | Texto + audio + durations + pitch + energy |
| **Velocidad** | Lenta (frame × frame) | Rápida (paralela) |
| **Control de melodía** | Implícito (atención) | Explícito (pitch predictor) |
| **Recomendación** | Empezar aquí | Implementar después |
