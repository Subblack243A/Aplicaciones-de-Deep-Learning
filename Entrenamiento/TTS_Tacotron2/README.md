# TTS - Tacotron 2 Sintético

Este directorio aísla la carga matemática de codificación acústica utilizada para sintetizar voz y canto (Texto a Voz).

### Arquitectura Técnica
Utilizamos una conjunción de dos pesos:
1. **Tacotron 2 (`models/tacotron2.py`)**: Dada una secuencia de texto (por ejemplo "Bury all your secrets"), aplica un algoritmo Attention Seq2Seq para predecir visualmente el espectrograma de las frecuencias de la letra.
2. **Vocoder - WaveRNN (`models/wavernn.py`)**: La imagen predictiva del paso anterior no emite sonido. El vocoder entra en acción infiriendo ondas sonoras de alta definición (22050 Hz) a partir de ese patrón espectro-gráfico.

Este proceso de aprendizaje toma recursos pesados, por lo que su `train.py` debe ser despachado preferiblemente a una GPU local en paralelo (`torch.device("cuda")`).
