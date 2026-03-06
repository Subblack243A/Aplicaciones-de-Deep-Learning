# Text Recognition Pipeline (CNN)

Este proyecto implementa una Red Neuronal Convolucional (CNN) desde cero utilizando **PyTorch** para clasificar imágenes de texto escrito en 4 categorías: `duvan`, `sierra`, `felipe` y `laura`.

## 📁 Estructura del Proyecto

```text
text-recognition/
├── data/               # Directorio para el dataset (organizado por carpetas de clase)
├── dataset.py          # Clase Custom Dataset y preprocesamiento de imágenes
├── model.py            # Definición de la arquitectura CNN
├── train.py            # Script principal de entrenamiento
├── predict.py          # Script de inferencia y guardado de resultados
├── model.pth           # Pesos del modelo guardados (se genera tras entrenar)
└── README.md           # Este archivo
```

## 🛠️ Requerimientos

Asegúrate de tener instalado Python 3.8+ y las siguientes librerías:

```bash
pip install torch torchvision pillow
```

_Nota: Se recomienda contar con soporte para CUDA si deseas aceleración por GPU._

## 🚀 Comandos de Ejecución

### 1. Preparación de Datos

Organiza tus imágenes en la carpeta `data/` siguiendo este esquema:

```text
data/
├── duvan/   --> imagen1.jpg, imagen2.png...
├── sierra/  --> ...
├── felipe/  --> ...
└── laura/   --> ...
```

### 2. Entrenamiento

Para iniciar el proceso de entrenamiento, ejecuta:

```bash
python train.py
```

El script realizará las siguientes acciones:

- Cargará y preprocesará las imágenes (redimensionado a 128x32 y normalización).
- Ejecutará el bucle de entrenamiento por el número de épocas configurado.
- Guardará los pesos finales en `model.pth`.

### 3. Inferencia (Predicción)

Para identificar una nueva imagen y guardar el resultado:

```bash
# Asegúrate de tener el archivo model.pth generado
python predict.py
```

_Nota: Puedes editar el script `predict.py` para cambiar la ruta de la imagen de prueba._

## 🔄 Flujo de Trabajo

1. **Preprocesamiento (`dataset.py`)**:
   - Redimensiona las imágenes a un tamaño fijo de **128x32 píxeles**.
   - Convierte las imágenes a escala de grises.
   - Normaliza los valores de los píxeles.
   - Mapea las etiquetas a índices: `{duvan: 0, sierra: 1, felipe: 2, laura: 3}`.

2. **Arquitectura (`model.py`)**:
   - Utiliza una CNN modular con 3 bloques de: `Convolución (3x3) -> ReLU -> MaxPool (2x2)`.
   - Incluye capas `Dropout` para reducir el sobreajuste.
   - Salida de 4 neuronas (_logits_) correspondientes a las clases.

3. **Inferencia y Salida (`predict.py`)**:
   - Carga el modelo entrenado (`model.pth`).
   - Identifica la clase de la imagen de entrada.
   - **Guarda el resultado en un archivo `.txt`** nombrado según la persona identificada (ej. `felipe.txt`).
