# Síntesis de Voz Cantada (SVS) - Tacotron 2

Este proyecto implementa un pipeline de síntesis de voz cantada basado en la arquitectura Tacotron 2. Convierte texto (letra) en un espectrograma Mel, que luego es reconstruido en audio utilizando el algoritmo Griffin-Lim.

## Características

- **Arquitectura Tacotron 2**: Modelo seq2seq con atención sensible a la ubicación (location-sensitive attention).
- **Tokenización Fonética**: Tokenización de caracteres simplificada basada en ARPAbet.
- **Procesamiento Mel**: Utilidad para extraer y reconstruir audio a partir de espectrogramas Mel.
- **Pipeline de Entrenamiento**: Bucle de entrenamiento completo con registro de pérdida (loss) y visualización de curvas de aprendizaje.
- **Pipeline de Inferencia**: Punto de entrada sencillo para generar canto a partir de texto.

## Requisitos

- Python 3.8+
- PyTorch
- Librosa
- Numpy
- Matplotlib
- Soundfile / Scipy

Puedes instalar los requisitos base usando:

```bash
pip install torch torchaudio numpy matplotlib librosa scipy soundfile phonemizer --break-system-packages
```

## Estructura del Proyecto

```
sing-activity/
├── main.py                # Punto de entrada (train/infer)
├── dataset.py             # Carga de datos
├── train.py               # Lógica de entrenamiento
├── inference.py           # Lógica de inferencia
├── models/
│   ├── tacotron2.py       # Arquitectura principal
│   └── wavernn.py         # Vocoder recurrente (opcional)
├── utils/
│   ├── text_utils.py       # Tokenización de texto
│   ├── audio_utils.py      # Procesamiento de audio
│   └── visualizer.py      # Visualización de métricas
└── dataset/               # Coloca tus archivos wav y transcripts.txt aquí
```

## Preparación del Dataset

El proyecto espera una carpeta `dataset/` con:

- `wavs/`: Subcarpeta que contiene archivos `.wav` (22050 Hz, mono).
- `transcripts.txt`: Un archivo donde cada línea tiene el formato: `audio_id|texto`.

## Uso

### Entrenamiento

Para entrenar el modelo desde cero:

```bash
python main.py --mode train --data_dir dataset --ckpt checkpoints/my_model.pt
```

### Inferencia

Para generar canto a partir de la letra:

```bash
python main.py --mode infer --text "Bury all your secrets in my skin" --ckpt checkpoints/my_model.pt --output output.wav
```

## Cómo funciona

1. **Codificación de Texto**: La letra se convierte en secuencias de IDs de caracteres.
2. **Encoder de Tacotron 2**: Aprende representaciones contextuales del texto.
3. **Atención**: Alinea la representación del texto con los frames del audio objetivo.
4. **Decoder de Tacotron 2**: Predice los frames del espectrograma Mel de forma autoregresiva.
5. **Post-Procesamiento**: Una PostNet refina el espectrograma generado.
6. **Vocoding**: Griffin-Lim convierte el espectrograma Mel refinado de nuevo en una forma de onda en el dominio del tiempo.
