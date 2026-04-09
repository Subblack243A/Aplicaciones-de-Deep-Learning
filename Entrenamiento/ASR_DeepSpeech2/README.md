# ASR - DeepSpeech2

Este directorio contiene la arquitectura de entrenamiento para la red neuronal iterativa enfocada en convertir *Voz a Texto* (Speech-to-Text).

### Componentes Internos
- **`dataset.py`**: Maneja el espectrograma (Mel-Spectrogram), extrae el audio y lo estandariza.
- **`model.py`**: Declara la estructura combinada que consta de capas **CNN 2D** y **RNN (Bi-GRU/LSTM)**.
- **`text_encoder.py`**: Convierte texto plano en arreglos numéricos matriciales decodificables.
- **`trainer.py`**: Ejecuta el loop de Backpropagation y evalúa iterativamente usando precisión CTC (Connectionist Temporal Classification).

> [!NOTE]
> La inferencia en producción (los despliegues con el modelo ya compilado) se ejecutan en `Implementaciones/Song_Activity/`. Aquí **solo se reentrena** el peso del modelo.
