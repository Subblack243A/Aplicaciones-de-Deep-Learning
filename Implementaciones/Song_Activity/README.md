# Song Activity - Implementación Completa

## Descripción

Este proyecto implementa el flujo completo de conversión de canción en inglés a español:
- **MP4 → WAV**: Convierte video a audio
- **WAV → STT**: Transcribe audio a texto usando ASR
- **STT → Traducción**: Traduce el texto al español
- **(Opcional) Texto → TTS**: Convierte texto traducido a audio

## Estructura

```
Song_Activity/
├── README.md
├── main.py              # Punto de entrada
└── pipeline.py          # Pipeline completo
```

## Uso

### Requisitos

Entrena primero el modelo ASR:
```bash
cd ../../Entrenamiento/ASR_DeepSpeech2
python main.py train --epochs 50
```

### Ejecutar

```bash
python main.py --input cancion.mp4 --output letra_espanol.txt
```

## Dependencias

- FFmpeg (para conversión de audio)
- TensorFlow (para ASR)
- Traductor (biblioteca de traducción)

Consulta `../../Entrenamiento/ASR_DeepSpeech2/README.md` para más detalles.