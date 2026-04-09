# Arquitectura Global y Guía Detallada de Reproducción

Este documento sirve como la fuente definitiva de la verdad para comprender la arquitectura interna, la comunicación entre micro-proyectos, el uso de redes neuronales y los requisitos técnicos para ejecutar el repositorio **Aplicaciones de Deep Learning**. 

Está diseñado para que cualquier ingeniero o compañero de equipo pueda entender qué hace cada archivo y cómo reproducir los resultados.

---

## 1. Topología del Proyecto

El proyecto está diseñado bajo un modelo arquitectónico de **Componentes Desacoplados**. Esto significa que la investigación y entrenamiento matemático de las redes están físicamente separados de su ejecución final (implementación).

* **`Entrenamiento/`**: El "Laboratorio". Aquí crecen los modelos matemáticos. No hay inferencias de usuario interactivas acá, solo scripts de optimización (Loss, Backpropagation, Datasets).
* **`Implementaciones/`**: El "Producto Final". Aquí se consumen los modelos exportados (o APIs de terceros) para lograr un flujo de extremo a extremo que el usuario final aprovechará con el mundo real (IoT, Audio, Kinect).
* **`Modelos/`**: El puente. Los archivos resultantes repuestos del `Entrenamiento/` (`.pth`, `.h5`, `.pt`) se guardan aquí para ser alimentados a las `Implementaciones/`.

---

## 2. Subsistemas de Entrenamiento (`Entrenamiento/`)

Aquí detallamos los núcleos de entrenamiento. Ninguno de estos consume recursos a menos que se invoque una época de entrenamiento.

### 2.1. Speech-to-Text (ASR_DeepSpeech2)
* **Basado en**: Arquitectura Baidu's DeepSpeech 2.
* **Objetivo**: Recibir un espectrograma de audio de voz y devolver carácteres (Texto).
* **Archivos Clave y Comunicación**:
  * `dataset.py` & `text_encoder.py`: Orquestan la carga de audio (WAV), creación de espectrogramas Mel y tokenización del texto a predecir.
  * `model.py`: Contiene las capas convolucionales 1D/2D más las capas recurrentes (RNN/GRU/LSTM) que capturan la estructura temporal de la voz.
  * `trainer.py` / `train_overfit.py`: Realizan el loop de entrenamiento. Computan la **CTC Loss** (Connectionist Temporal Classification) para alinear la voz con el texto sin saber dónde empieza una vocal.
* **Consumo de Hardware**: Alto. Requiere GPU con al menos 8GB+ de VRAM para batches decentes.

### 2.2. Síntesis de Canto (TTS_Tacotron2)
* **Basado en**: Arquitectura Tacotron 2 (Google) + Vocoder WaveRNN.
* **Objetivo**: Recibir una letra de canción y devolver un archivo de audio WAV de voz cantada.
* **Archivos Clave y Comunicación**:
  * `models/tacotron2.py`: Arquitectura seq2seq con Atención. Convierte la secuencia de texto en un "Mel-Espectrograma".
  * `models/wavernn.py`: Es un Vocoder. Tacotron genera una "imagen" de audio, pero WaveRNN convierte esa imagen sintética en una forma de onda de audio de alta fidelidad que podemos escuchar.
  * `utils/`: Tienen tareas de limpieza, como normalización e iteración gráfica del audio (`visualizer.py`, `trim_vocals.py`).
  * `train.py`: Entrena ambos modelos.
* **Consumo de Hardware**: Muy Alto. El entrenamiento de algoritmos TTS es costoso, requiere GPUs sólidas (NVIDIA serie 30XX/40XX o equivalentes en la nube).

### 2.3. Generación Imagen (T2I_Text2Image)
* **Objetivo**: Generar representaciones visuales condicionados por descripción de texto.
* **Archivos Clave**:
  * `train_text2image.py`: Este es un modelo híbrido escrito a medida. Usa un `TextEncoder` basado en bloques **TransformerEncoder** para asimilar sintaxis del texto y embeber el lenguaje.
  * Luego, utiliza un `ImageDecoder` con capas **ConvTranspose2d** (Deconvolución) combinadas con **Cross-Attention** (MultiheadAttention). El ruido aleatorio actúa como semilla, y gracias a la atención cruzada, la creación visual sigue el hilo conceptual del texto.
  * Usa métricas perceptuales avanzadas como LPIPS (Learned Perceptual Image Patch Similarity) y SSIM integrado para que la "Loss" sea fiel al ojo humano y no sólo píxel a píxel puro. Añade **Captum** para explicabilidad e ingeniería inversa sobre qué impacto visual tiene cada palabra.
* **Consumo de Hardware**: Depende de la escala; usa entrenamiento de precisión mixta (`torch.amp.autocast`), un estándar sólido para reducir consumo a casi la mitad de VRAM.

### 2.4. Optical Character Recognition (OCR_EasyOCR)
* **Basado en**: EasyOCR / CRNN (CNN + RNN + CTC).
* **Funcionamiento**: Extrae las características puras espaciales de los trazos (Pixeles en una imagen de Kinect o Manito), la secuencia de curvas es leída en orden recurrente, decodificando las letras probables.

---

## 3. Implementaciones de Extremo a Extremo (`Implementaciones/`)

Aquí el flujo de trabajo cobra vida.

### 3.1. Song Activity
* **Flujo**: MP4 → WAV → Separación Vocal → STT → Traducción → (Opcional) TTS Cantado.
* **Mecánica Archivos**:
  * `mp4_to_mp3_converter.py` extrae el canal sonoro usando **FFmpeg**, luego `audio_processor.py` hace limpieza pasabanda/hpss.
  * `predictor.py` invoca los pesos de `ASR_DeepSpeech2` para escanear el audio y devolver texto (típicamente Inglés).
  * (Opcional) Usando la bandera `--sing`, se invoca iterativamente a `tts_singer.py`, el cual carga directamente de forma enrutada el modelo *Tacotron2* para cantar el output de texto.
* **Consumo**: Variable. Transcribir es rápido, pero adjuntarle la síntesis vocal requiere computación GPU o varios minutos en CPU por estrofa.

### 3.3. Escritura Aire Kinect
* **Flujo**: Visión Espacial Microsoft Kinect → Trazado Gráfico 2D → OCR → TTS Voz humana.
* **Mecánica Archivos**:
  * `pykinect_v1/`: Un Wrapper wrapper / Interfaz C Types (NUI) para dialogar con el SDK privativo de Windows/Kinect. Permite adquirir mapas de esqueleto (Skeleton tracking) y profundidad 3D.
  * `kinect_app_v2.py`: Rastreando las coordenadas del punto central de la mano del usuario, pinta mediante OpenCV en un "lienzo virtual" píxeles continuos asemejando un lápiz.
  * Una vez la persona frena la escritura, la imagen recortada entra al modelo de OCR, quien tira un string (`"Hola Mundo"`), enviándolo entonces por TTS para feedback audible final.

### 3.4. Manito (Proyecto Robótico Hardware-in-The-Loop)
* **Flujo**: Computadora PC → T2I imagen literal → OCR reconocimiento de letra → Intérprete → Arduino/ESP32 C6 Motores PWM.
* **Mecánica Archivos**:
  * `enviar_letra.py`: Escucha el resultado cognitivo del modelo local en la Python App. Hace requests o abre un puerto Serial (COM/TTY).
  * `config.json`: Un diccionario fundamental que mapea una letra (ej. 'A') a grados específicos `(0-180)` de tensión para cada uno de los 5 tendones (los 5 servomotores).
  * `ESP32_C6/src/main.cpp`: Firmware de microcontrolador IoT programado en C++ (PlatformIO). Lee la letra/json e impulsa usando librerías de temporizador (Timmers) de hardware los pines PWM para que los dedos de la mano robótica bajen y suban cerrando un lenguaje de señas en el mundo físico.

---

## 4. Guía de Reproducción Técnica y Pasos

Instrucciones para que un colega se acople al flujo.

### Paso 1: Configurar el Entorno

Para evitar que se formen agujeros negros de dependencias en las librerías nativas o que haya coalición entre versiones de CUDA, deben usar **Anaconda/Miniconda** o entornos virtuales:

```bash
# Asumiendo que se clona el repo "Aplicaciones-de-Deep-Learning"
cd Aplicaciones-de-Deep-Learning

# Instalar requisitos de hardware de fondo primero:
# (Ubuntu/Linux)
sudo apt update && sudo apt install ffmpeg libsndfile1 build-essential
```

### Paso 2: Reproducir Entrenamientos

Se recomienda al equipo usar las carpetas `Entrenamiento` si van a refinar los parámetros matemáticos.
* Los script aceptan el hardware dinámico (`device = torch.device("cuda" ...)`): si tienen GPU de Nvidia con drivers bien puestos, irá a CUDA, de lo contrario fallback a CPU.
* **Reproducir T2I (Text-to-Image)**:
  ```bash
  cd Entrenamiento/T2I_Text2Image
  pip install torch torchvision torchmetrics lpips captum datasets kaggle matplotlib

  # 1. Configurar su token en ~/.kaggle/kaggle.json
  # 2. Descargar datos experimentales ejecutando una prueba
  python descargar_dataset.py

  # 3. Arrancar ciclos. Presten atención a "Early stopping activado".
  python train_text2image.py
  ```

### Paso 3: Reproducir Implementaciones IoT "Manito"

Para la fase de robótica, tu compañero necesita lo siguiente en Hardware y Software respectivo:

* IDE: **Visual Studio Code** con la extensión `PlatformIO`.
* Target Device: Placa **ESP32-C6**.
* Componentes: Modulo PWM, 5 Servos analógicos alimentados desde una fuente de corriente independiente de 5V (No del pin USB 3.3V).

**Procedimiento en PC**:
1. Abrir Vscode y abrir folder en `Implementaciones/Manito/ESP32_C6`.
2. Conectar la placa ESP32 por USB C. Apretar el icono de "Upload/Subir" en la parte baja de PlatformIO. Compilará C++ y se subirá.
3. Ejecutar `python enviar_letra.py` en la máquina local para iniciar las orquestaciones de comunicación y empezar a mover la mano.

---

## Resumen de Buenas Prácticas y Modelado Involucrado
* **Modularidad Física y Lógica**: En lugar de tener el hardware IoT pegado a la inteligencia abstracta, las implementaciones comunican mediante `config.json` y protocolos seriales (Interfaces estándar).
* **Early Stopping System**: La librería personalizada frena el descenso estocástico en entrenamiento si la red satura y empieza a hacer Overfitting por 10 épocas (`patience=10`), un estándar élite.
* **Métricas de Percepción Profunda**: Usar LPIPS por encima de MSE solitario hace que las redes generativas engañen a características perceptuales en redes VGG en lugar de a cálculos vectoriales cuadrados, maximizando la belleza de salida sobre la precisión por píxel absoluto.
