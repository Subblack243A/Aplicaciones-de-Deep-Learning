# Song Activity

Este proyecto consiste en la creación de una aplicación que permita transformar una canción en inglés a español.

## 1. Estructura del proyecto

- `mp4_to_mp3_converter.py`: Convierte un archivo MP4 a MP3 (Implementación Manual con `ffmpeg-subprocess`).
- `manual_audio.py`: Módulo de procesamiento manual (HPSS y Filtros Mel) usando NumPy.
- `mp3_to_text.py`: Convierte un archivo MP3 a texto.
- `translate_text.py`: Traduce un archivo de texto a español.
- `text_to_audio.py`: Convierte un archivo de texto a audio.
- `main.py`: Programa principal que orquesta la ejecución del pipeline (Video -> MP3 -> Voz Separada -> Espectrograma).

## 2. Requerimientos

Este proyecto está siendo ejecutado en `Python 3.14`.

Para una correcta ejecución del programa se necesitará tener en cuenta las siguientes librerías:

- `numpy`, `scipy`: Para cálculos matemáticos (HPSS, Filtros Mel).
- `librosa`, `soundfile`: Para carga/guardado de audio y STFT base.
- `matplotlib`: Para visualizar espectrogramas.
- `ffmpeg`: Ejecutable del sistema (debe estar en el PATH).

Para instalar todas estas librerías solo es necesario correr el siguiente comando:

```bash
pip install -r song-activity/requirements.txt
```

---

## 3. Documentación Técnica: Procesamiento de Audio Manual

A continuación se detalla la implementación realizada "desde cero" para la manipulación de audio.

### Fundamentos Matemáticos

#### Separación Armónica-Percusiva (HPSS)

El algoritmo se basa en la diferente morfología de los sonidos en un espectrograma:

- **Sonidos Armónicos (Voz)**: Líneas horizontales (estables en el tiempo).
- **Sonidos Percusivos (Ritmo)**: Líneas verticales (transitorios de banda ancha).

**Proceso:**

1. **STFT**: Transformada de Fourier de Tiempo Reducido.
    $$D(t, f) = STFT(x)$$
2. **Filtros de Mediana**:
    - **Horizontal ($H$)**: Resalta estructuras armónicas.
        $$H(t, f) = \text{median}(|D(t, f)|, \text{kernel}_{time})$$
    - **Vertical ($P$)**: Resalta estructuras percusivas.
        $$P(t, f) = \text{median}(|D(t, f)|, \text{kernel}_{freq})$$
3. **Enmascaramiento Suave (Soft Masking / Wiener Filter)**:
    Originalmente usamos máscaras binarias (Hard Masking), pero generaban distorsión. Para mejorar la calidad, implementamos **Soft Masking**:

    $$M_h(t, f) = \frac{H(t, f)}{H(t, f) + P(t, f)}$$
    $$M_p(t, f) = \frac{P(t, f)}{H(t, f) + P(t, f)}$$

    Esto permite que una frecuencia pertenezca "parcialmente" a ambas fuentes, suavizando el sonido resultante.

4. **Reconstrucción**:
    $$x_{armónico} = iSTFT(M_h \cdot D)$$

#### Espectrograma de Mel

Transformación de frecuencia (Hz) a la escala de percepción humana (Mel).

**Fórmula de Conversión:**
$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**Construcción del Banco de Filtros:**
Se crea una matriz de filtros triangulares que mapean los bins de la FFT a bandas Mel:

- Se definen puntos equidistantes en el dominio Mel.
- Se convierten de nuevo a Hz y luego a índices de FFT.
- Se crean triángulos con peso 1 en el centro y 0 en los extremos.

$$S_{mel} = \log(W_{mel} \cdot |STFT|^2)$$

---

## 4. Guía de Uso del Pipeline

El script `main.py` orquesta todo el proceso para preparar el audio para el modelo de Voz a Texto.

**Ejecución:**

```bash
python3 song-activity/main.py video_entrada.mp4
```

**Salidas:**

- `video_entrada.mp3`: Audio extraído.
- `processed_voice.wav`: Componente armónica (voz) separada -> **Input para STT**.
- `processed_background.wav`: Componente percusiva/fondo.
- `voice_mel_spectrogram.png`: Imagen del espectrograma.
