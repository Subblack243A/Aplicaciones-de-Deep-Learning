# Conversión de Audio (MP4 → WAV)

Módulo responsable de extraer y convertir audio desde archivos multimedia a formato WAV mono, listo para procesamiento.

## Archivo Principal

- **`audio_converter.py`** → Clase `AudioConverter`

## Funcionalidades

| Método | Descripción |
|--------|-------------|
| `mp4_to_mp3(input, output)` | Convierte MP4 a MP3 usando FFMPEG |
| `mp3_to_wav(input, output, sr)` | Convierte MP3 a WAV mono |
| `mp4_to_wav(input, sr)` | Conversión directa MP4 → WAV |
| `any_to_wav(input, sr)` | Auto-detecta formato y convierte a WAV |

## Uso

```python
from audio_converter import AudioConverter

# Convertir cualquier formato a WAV (16kHz, mono)
wav_path = AudioConverter.any_to_wav("cancion.mp4", sample_rate=16000)
```

## Requisitos

```bash
# FFMPEG debe estar instalado en el sistema
sudo apt-get install ffmpeg    # Ubuntu/Debian
brew install ffmpeg            # macOS
```

## Notas Técnicas

- **Sample rate por defecto**: 16000 Hz (óptimo para ASR)
- **Para SVS**: Usar 22050 Hz (`sample_rate=22050`)
- Archivos intermedios (MP3) se eliminan automáticamente
- Soporta: `.mp4`, `.mp3`, `.wav`, `.avi`, `.mkv`, `.mov`, `.webm`
