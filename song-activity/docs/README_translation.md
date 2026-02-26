# Traducción (Translation)

Traducción automática de texto usando `argostranslate` (modelo local, sin API externa).

## Archivo

- **`traduction.py`** → Clase `TranslationService`

## Funcionalidades

| Método | Descripción |
|--------|-------------|
| `translate_text(text)` | Traduce un string |
| `translate_file(input, output)` | Traduce un archivo completo en batches paralelos |

## Uso como módulo

```python
from traduction import TranslationService

translator = TranslationService(from_code="en", to_code="es")

# Texto individual
spanish = translator.translate_text("Bury all your secrets in my skin")

# Archivo completo
translator.translate_file("transcription.txt", "translated.txt", batch_size=5)
```

## Uso desde CLI

La traducción se ejecuta automáticamente como parte del comando `predict`:

```bash
python main.py predict \
    --model_path ./model_saved/final_model.pt \
    --input cancion.mp4 \
    --output ./output/transcription.txt \
    --target_lang es    # Idioma destino (es, fr, de, etc.)
```

Para deshabilitar la traducción: `--target_lang none`

## Idiomas Soportados

Los paquetes de idioma se descargan automáticamente la primera vez. Algunos pares disponibles:

| De | A | Código |
|----|---|--------|
| English | Español | `en` → `es` |
| English | Français | `en` → `fr` |
| English | Deutsch | `en` → `de` |
| English | Português | `en` → `pt` |

## Notas Técnicas

- El modelo de traducción se ejecuta **100% local** (sin internet después de la primera descarga)
- Usa procesamiento paralelo (ThreadPoolExecutor) para archivos largos
- Batch size configurable para optimizar velocidad vs memoria
