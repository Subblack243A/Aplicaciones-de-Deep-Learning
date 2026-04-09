# Aplicaciones de Deep Learning

Este repositorio contiene proyectos de Deep Learning organizados sistemáticamente en tres secciones para garantizar la modularidad y escalabilidad:
- **Entrenamiento**: Código de acceso, dataset, y entrenamiento para cada red neuronal por separado.
- **Implementaciones**: Proyectos o despliegues completos que orquestan uno o múltiples modelos para cumplir un flujo funcional.
- **Modelos**: Carpeta para guardar los modelos preentrenados listos para su uso.

---

## Estructura del Proyecto

```
Aplicaciones-de-Deep-Learning/
├── Entrenamiento/           # Códigos de entrenamiento de cada modelo base
│   ├── ASR_DeepSpeech2/    # Reconocimiento de voz (Speech-to-Text)
│   ├── T2I_Text2Image/     # Difusión / Texto a Imagen
│   ├── TTS_Tacotron2/      # Síntesis de voz cantada (Text-to-Speech)
│   └── OCR_EasyOCR/        # OCR (Adaptado según Kinect app)
│
├── Implementaciones/        # Proyectos (Flujos de trabajo completos)
│   ├── Song_Activity/      # MP4 → WAV → STT → Traducción → (Opcional) TTS
│   ├── Escritura_Aire_Kinect/ # Escritura en aire con Kinect → OCR → TTS
│   └── Manito/             # Texto → Imagen → OCR → ESP32_C6 (Mano robótica)
│
└── Modelos/                # Archivos `.pth` y resultados listos para inferencia
```

---

## Entrenamiento (Redes Base)

Cada arquitectura mantiene aislado su ciclo de experimentación.

| Red / Arquitectura | Descripción | Ubicación |
|-----|-------------|-----------|
| **ASR (DeepSpeech2)** | Speech-to-Text / STT | `Entrenamiento/ASR_DeepSpeech2/` |
| **T2I (Text-to-Image)** | Texto a Imagen | `Entrenamiento/T2I_Text2Image/` |
| **TTS (Tacotron2)** | Síntesis de voz cantada | `Entrenamiento/TTS_Tacotron2/` |
| **OCR** | Optical Character Recognition | `Entrenamiento/OCR_EasyOCR/` |

---

## Implementaciones Funcionales

### 1. Song Activity
Proceso completo unificado: **MP4 → WAV → STT → Traducción → (Opcional) TTS Cantado**

1. Convierte archivo de video (MP4) a audio (WAV).
2. Transcribe el audio a texto usando ASR.
3. Traduce el texto al español.
4. Si se especifica el flag `--sing`, convierte la transcripción directamente a voz cantada sintetizada usando sub-componentes de Tacotron2.

- **Ubicación:** `Implementaciones/Song_Activity/`

### 3. Escritura Aire Kinect
Proceso: **Kinect → Escritura Visual → OCR → TTS**

1. Utiliza el sensor Kinect para rastrear la mano y escribir gráficamente en el aire.
2. Digitaliza los trazos a texto legible usando OCR.
3. Convierte el texto reconocido a voz (TTS) para lectura de los trazos.

- **Ubicación:** `Implementaciones/Escritura_Aire_Kinect/`

### 4. Manito (Mano Robótica Autónoma)
Proceso: **Texto Visual → Imagen → OCR → Traducción y MCU (ESP32)**

1. Se ingresa el texto directamente desde la PC.
2. El input se mapea y convierte internamente de texto a imagen representativa.
3. Se emplea inferencia OCR para procesar de regreso a caracteres de control.
4. Se envían los tokens generados a los servomotores a través de la ESP32_C6 por serial/red.

- **Ubicación:** `Implementaciones/Manito/`

---

## Cómo Usar y Desplegar

### Fase de Entrenamiento

Al querer mejorar o alterar una red, diríjase a su directorio en la carpeta de `Entrenamiento/`:

```bash
# Entrenar ASR (DeepSpeech2)
cd Entrenamiento/ASR_DeepSpeech2
python trainer.py --epochs 50

# Entrenar TTS (Tacotron2)
cd Entrenamiento/TTS_Tacotron2
python train.py

# Entrenar T2I (Text-to-Image)
cd Entrenamiento/T2I_Text2Image
python train_text2image.py
```

### Ejecuciones e Inferencia

Cada proyecto modular bajo `Implementaciones/` contiene sus propios scripts aislados para manejar las sub-etapas como la comunicación serial y la interfaz final. Consulte el respectivo `README.md` al interior de esa implementación para detalles de uso.

## Requisitos Generales

- Python 3.8+
- PyTorch / TensorFlow (Depende de qué implementación utilice).
- FFmpeg (Requerido crucialmente para el procesamiento y particionado de audios y videos).