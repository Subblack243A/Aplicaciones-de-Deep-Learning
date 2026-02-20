# Song Activity (Reconocimiento Automático de Voz)

Este proyecto consiste en la creación de una aplicación basada en un modelo de Reconocimiento Automático de Voz (ASR, por sus siglas en inglés) capaz de transcribir palabras de una canción o archivo de audio/video. El modelo se diseña y entrena localmente utilizando TensorFlow y procesa los datos en formato de espectrograma temporal.

---

## 1. Arquitectura y Funcionamiento

El modelo de transcripción está basado en la arquitectura **DeepSpeech 2**. El flujo de datos y funcionamiento general se divide en varias etapas:

1. **Preprocesamiento de Audio (`AudioProcessor`)**: El archivo de entrada (audio o video convertido a `.wav` a través de `ffmpeg`) es cargado y analizado acústicamente con la librería `librosa`. Se extrae su [espectrograma Mel](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), el cual es normalizado (media 0 y desviación estándar 1) para ser presentado al modelo en forma de matriz bidimensional (tiempo $\times$ frecuencia).
2. **Modelo Acústico (`ASRModel`)**:
   - **CNN (Red Neuronal Convolucional):** Actúan sobre el espectrograma filtrando las características locales (frecuencias y cambios rápidos).
   - **Bi-LSTM (Memoria a Corto y Largo Plazo Bidireccional):** Estas capas recurrentes procesan la secuencia hacia adelante y hacia atrás, capturando las dependencias temporales y el contexto del habla.
   - **Capa Densa:** Proyecta el estado final de las RNN en probabilidades sobre los caracteres del vocabulario más un token adicional llamado _blank token_.
3. **Pérdida y Decodificación (CTC y `TextEncoder`)**: Utiliza la **función de pérdida CTC (Connectionist Temporal Classification)** que permite entrenar modelos de secuencias cuando la entrada (audio) y la salida (texto) no están perfectamente alineadas en el tiempo. Para generar el texto final desde las predicciones, el sistema utiliza enfoques _Greedy Decoding_ o _Beam Search_ para eliminar predicciones redundantes y _blank tokens_.

---

## 2. Estructura del Proyecto

El código está modularizado en clases especializadas para cada parte del _pipeline_:

- `main.py`: Punto de entrada CLI (Command Line Interface). Orquesta la ejecución para entrenar o predecir basándose en argumentos por consola.
- `model.py`: Define la topología construida en Keras/TensorFlow para el ASR (`ASRModel`) y la capa personalizada de pérdida (`CTCLayer`).
- `trainer.py`: Clase `Trainer` que implementa un loop de entrenamiento personalizado con `tf.GradientTape`, cálculo de métricas en validación, graficado de la convergencia y guardado estratégico (checkpoints) del modelo.
- `predictor.py`: Clase `Predictor` capaz de cargar un modelo guardado en formato `.keras` y procesar tanto audios como videos pasándolos por el decodificador para dar la inferencia en texto.
- `dataset.py`: Contiene `LibriSpeechDataset`, encargado de descargar el conjunto de datos, extraerlo, procesar en disco los `.flac` con sus transcripciones, y orquestar el pipeline ultra-eficiente a través de `tf.data.Dataset`.
- `audio_processor.py`: Funciones para cargar `.wav`, convertir a espectrogramas Mel normalizados y estructurar la matriz para que coincida con las dimensiones de entrada del tensor de la red neuronal.
- `audio_converter.py`: Wrapper alrededor de `ffmpeg-python` que se encarga de extraer el audio temporal en `.wav` (a $16,000$ Hz) cuando se le pasa un formato de video como `.mp4` u otros de audio comprimido (`.mp3`).
- `text_encoder.py`: Gestiona un vocabulario delimitado (letras `a-z` y el espacio vacio), mapeándolo de texto a números enteros bidireccionalmente. Contiene lógica para lidiar con el espacio "blank" del CTC.

---

## 3. Consideraciones más Importantes

1. **Dependencia FFMPEG**: Toda transformación y carga de formato cruzado dependerá de tu sistema subyacente interactuando con FFmpeg. Asegúrate de instalar `ffmpeg` globalmente en tu PC (vía `brew install ffmpeg` en macOS o `apt-get install ffmpeg` en Ubuntu).
2. **Uso de Memoria y Dataset**: El entrenamiento por defecto descarga partes clave de **LibriSpeech**. Este dataset consume espacio en disco y requiere bastante poder de cómputo en la etapa final de ensamblaje del `tf.data`. Si entrenas desde 0 en tu ordenador local sin GPU potente, considera emplear splits más pequeños y disminuir drásticamente el tamaño del `batch_size`.
3. **Instalación de requisitos vía Conda**: Para un manejo limpio del sistema y de los drivers integrados en la tarjeta gráfica de tu entorno para TF (TensorFlow), es muy recomendable utilizar Conda. Tienes a disposición `environment.yml` (para entornos genéricos).

---

## 4. Cómo usar la herramienta

### Instalación del Entorno

Es recomendado crear un entorno virtual para instalar todas las dependencias.

Con **pip**:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Con **Conda** (Especialmente recomendado para tener aceleración GPU correcta):

```bash
conda env create -f environment.yml
conda activate song-activity
```

_(Nota para usuarios de Mac: Puedes usar `environment-mac.yml` si requieres configuraciones en arquitecturas ARM como M1/M2/M3)._

---

### Comandos de uso principal (`main.py`)

La herramienta se controla desde la línea de comandos usando dos _subcomandos_ primarios: **train** y **predict**.

#### 1. Entrenar (`train`)

Este comando descarga los datos, prepara los tensores, inicializa la topología DeepSpeech 2 y arranca el entrenamiento.

```bash
python main.py train \
    --data_dir ./data \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --save_path ./model_saved
```

**Parámetros:**

- `--data_dir`: Ruta donde se descargará LibriSpeech o en la que ya se encuentra si es que lo descargaste previamente. (Por defecto `./data`)
- `--epochs`: Máximo de épocas de aprendizaje. (Por defecto `50`)
- `--batch_size`: Tamaño del lote. Redúcelo si enfrentas errores de Over Flow (OOM). (Por defecto `16`)
- `--learning_rate`: Tasa de corrección (LR) del modelo. (Por defecto `0.001`)
- `--save_path`: Carpeta sobre la cual guardar los pesos exportados una vez finalice. (Por defecto `./model_saved`)

#### 2. Predecir / Transcribir (`predict`)

Una vez tengas un modelo entrenado, o si descargaste algún peso en `.keras` existente en la ruta `model_saved`, usa esta instrucción pasándole el archivo objetivo:

```bash
python main.py predict \
    --model_path ./model_saved \
    --input mi_cancion.mp4
```

**Opcionales para la predicción:**

- `--beam_width`: Activa Beam Search y le indica su "ancho" en lugar del tradicional "Greedy Decoder". (Ej. `--beam_width 10`). Ayuda a explorar múltiples secuencias viables reduciendo la tasa de errores.
