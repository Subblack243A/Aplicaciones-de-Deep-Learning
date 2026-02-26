# Guía de Entrenamiento TTS (GPU NVIDIA)

Guía para descargar el dataset, entrenar **Tacotron 2** y **FastSpeech 2**, y comparar ambos modelos.

---

## Requisitos

| Requisito | Mínimo | Recomendado |
| --------- | ------ | ----------- |
| **GPU** | NVIDIA 4GB VRAM | NVIDIA 8GB+ VRAM |
| **CUDA** | 11.8+ | 12.x |
| **Python** | 3.10 | 3.11 |
| **RAM** | 8 GB | 16 GB |
| **Disco** | 10 GB | 20 GB |

---

## Opción A: Todo Automatizado

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Ejecutar pipeline completo
chmod +x train_pipeline.sh
./train_pipeline.sh
```

**Eso es todo.** El script descarga LJSpeech, entrena ambos modelos, y genera audios de prueba.

Para test rápido (~30 min): editar `train_pipeline.sh` línea 16 → `MAX_SAMPLES="500"`

---

## Opción B: Paso a Paso

### 1. Instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate

# PyTorch con CUDA 12.x
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Resto de dependencias
pip install -r requirements.txt
```

### 2. Verificar CUDA

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. Descargar dataset LJSpeech (~2.6 GB)

```bash
# Descargar + preparar (incluye extracción de pitch, energy, durations)
python3 download_tts_dataset.py --output_dir ./data --dataset_dir ./dataset

# Para test rápido con menos datos:
python3 download_tts_dataset.py --output_dir ./data --dataset_dir ./dataset --max_samples 500

# Si solo vas a entrenar Tacotron 2 (no necesita features):
python3 download_tts_dataset.py --output_dir ./data --dataset_dir ./dataset --skip_features
```

### 4. Entrenar Tacotron 2

```bash
python3 main.py train-svs \
    --model tacotron2 \
    --dataset_dir ./dataset \
    --epochs 500 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --save_path ./svs_saved
```

### 5. Entrenar FastSpeech 2

```bash
python3 main.py train-svs \
    --model fastspeech2 \
    --dataset_dir ./dataset \
    --epochs 600 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --save_path ./svs_saved
```

### 6. Sintetizar con ambos

```bash
# Tacotron 2
python3 main.py synthesize \
    --model_type tacotron2 \
    --model_path ./svs_saved/tacotron2/final_model.pt \
    --input "Bury all your secrets in my skin" \
    --output ./output/test_tacotron2.wav

# FastSpeech 2
python3 main.py synthesize \
    --model_type fastspeech2 \
    --model_path ./svs_saved/fastspeech2/final_model.pt \
    --input "Bury all your secrets in my skin" \
    --output ./output/test_fastspeech2.wav
```

---

## Comparación de Resultados

Después del entrenamiento, revisar estos archivos:

| Archivo | Tacotron 2 | FastSpeech 2 |
| ------- | ---------- | ------------ |
| Training report | `svs_saved/tacotron2/training_report.md` | `svs_saved/fastspeech2/training_report.md` |
| Learning curves | `svs_saved/tacotron2/learning_curves.png` | `svs_saved/fastspeech2/learning_curves.png` |
| Métricas JSON | `svs_saved/tacotron2/metrics.json` | `svs_saved/fastspeech2/metrics.json` |
| Audio de prueba | `output/test_tacotron2.wav` | `output/test_fastspeech2.wav` |
| Mel spectrogram | `output/test_tacotron2_mel.png` | `output/test_fastspeech2_mel.png` |

### Criterios para decidir

| Criterio | Tacotron 2 | FastSpeech 2 |
| -------- | ---------- | ------------ |
| Naturalidad del audio | Mejor | Buena |
| Velocidad de síntesis | Lenta | Rápida (paralela) |
| Velocidad de entrenamiento | Más lenta | Más rápida |
| Control de melodía | Implícito | Explícito (pitch/energy) |
| Estabilidad | Puede fallar | Más estable |

---

## Ajuste según GPU

| VRAM | `batch_size` | `max_samples` | Tiempo por modelo |
| ---- | ------------ | ------------- | ----------------- |
| 4 GB | 8 | 500 | ~1 hora |
| 6 GB | 16 | 2000 | ~2 horas |
| 8 GB+ | 32 | todos | ~3-4 horas |

---

## Troubleshooting

**CUDA no detectado** → `nvidia-smi` para verificar driver
**Out of Memory** → Reducir `--batch_size` a 8
**Descarga lenta** → Descargar LJSpeech manualmente desde <https://keithito.com/LJ-Speech-Dataset/> y extraer en `./data/`
