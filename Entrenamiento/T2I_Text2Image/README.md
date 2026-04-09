# Generador Text-to-Image (T2I) — Entrenamiento

Sistema de generación de imágenes con texto legible sobre fondos sólidos. 
Arquitectura **Transformer Encoder-Decoder** con Cross-Attention multi-resolución, implementada desde cero en PyTorch.

---

## Estructura (3 archivos Python)

```
T2I_Text2Image/
├── model.py          # TextEncoder + ImageDecoder + Text2ImageModel (todo el modelo)
├── dataset.py        # Generador sintético + DataLoader + tokenización
├── train.py          # Entrenamiento completo: métricas, early stopping, plots, XAI
├── fonts/            # Fuentes TTF (DejaVuSans)
├── data/             # Dataset sintético (auto-generado)
├── outputs/          # Checkpoints, gráficas, mapas XAI
├── requirements.txt
├── environment.yml
├── .env
└── README.md
```

---

## Arquitectura del Modelo

```
Input "Santiago" → Tokenización (char→ASCII ID)
       ↓
TextEncoder (4 capas Transformer, 8 heads, PositionalEncoding)
       ↓ text_features
noise z~N(0,1) + text_features → ImageDecoder
       ↓
  Block1: 8→16  + CrossAttention
  Block2: 16→32 + CrossAttention  
  Block3: 32→64 + CrossAttention
  Conv2d → Tanh
       ↓
Output: 64×64 RGB imagen
```

## Métricas

| Métrica | Tipo | Propósito |
|---------|------|-----------|
| **MSE** | Loss (↓) | Reconstrucción píxel a píxel |
| **SSIM** | Monitor (↑) | Similitud estructural / legibilidad |
| **LPIPS** | Loss (↓) | Coherencia perceptual (VGG) |

**Training Loss** = MSE + 0.1 × LPIPS

**Early Stopping**: patience=10, min_delta=0.001, max 200 épocas.

---

## Reproducción

```bash
# 1. Entorno
conda env create -f environment.yml && conda activate t2i-env
# O: pip install -r requirements.txt

# 2. Generar dataset (5000 pares texto→imagen)
python train.py --generate-data-only

# 3. Entrenar
python train.py                    # Completo (max 200 épocas)
python train.py --max-epochs 5     # Smoke test
python train.py --resume           # Continuar desde checkpoint
```

El modelo entrenado se guarda automáticamente en `../../Modelos/t2i_text2image.pth`.

---

## Flujo con Manito

Consultar `Implementaciones/Manito/pipeline_t2i.py` para el flujo completo:

```
Texto → T2I (este modelo) → Imagen → OCR (EasyOCR) → Letras → ESP32 → Mano
```

La implementación incluye un **fallback oculto** (Pillow directo) que se activa automáticamente si el modelo no está disponible.
