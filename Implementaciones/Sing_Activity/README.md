# Sing Activity - Implementación Completa

## Descripción

Este proyecto convierte texto (letra de canción) en audio cantado usando Tacotron2.

## Estructura

```
Sing_Activity/
├── README.md
├── main.py              # Punto de entrada
├── pipeline.py          # Pipeline completo
└── utils/              # Utilidades
```

## Uso

### Requisitos

Entrena primero el modelo TTS:
```bash
cd ../../Entrenamiento/TTS_Tacotron2
python main.py --mode train --data_dir ./dataset
```

### Ejecutar

```bash
# Usar archivo de letra
python main.py --mode infer --text_file letra.txt --output mi_cancion

# Usar texto directo
python main.py --mode infer --text "Hello world" --output saludo
```

## Dependencias

- PyTorch
- Librosa
- Matplotlib

Consulta `../../Entrenamiento/TTS_Tacotron2/README.md` para más detalles.