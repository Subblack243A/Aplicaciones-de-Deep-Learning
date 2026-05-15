"""
T2I Module — Text-to-Image.
Intenta usar un modelo de difusión (diffusers) si está disponible.
Si no, usa fallback de Pillow para generar imágenes estilizadas del texto.

Variable de entorno T2I_MODE=pillow|diffusion para forzar modo.
"""
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config

# ── Paletas y fuentes para fallback Pillow ────────────────────────────
_PALETTES = [
    ((255, 255, 255), (20, 20, 20)),
    ((245, 245, 245), (50, 50, 70)),
    ((0, 0, 0), (240, 240, 240)),
    ((30, 30, 60), (220, 220, 255)),
    ((50, 50, 80), (255, 215, 0)),
    ((250, 248, 240), (60, 60, 80)),
    ((232, 245, 233), (27, 94, 32)),
    ((255, 243, 224), (230, 81, 0)),
]

_FONT_CANDIDATES = [
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
]


def _find_font(size):
    for p in _FONT_CANDIDATES:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


def _pillow_render(text: str, img_size: int = 256, seed: int = 42) -> Image.Image:
    """Genera una imagen estilizada con el texto usando Pillow."""
    random.seed(seed)
    bg_color, fg_color = random.choice(_PALETTES)

    img = Image.new("RGB", (img_size, img_size), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Dibujar un degradado sutil o patrón de fondo
    for i in range(img_size):
        alpha = int(255 * (1 - abs(i - img_size // 2) / (img_size // 2)) * 0.05)
        color = (max(0, bg_color[0] - alpha), max(0, bg_color[1] - alpha), max(0, bg_color[2] - alpha))
        draw.line([(0, i), (img_size, i)], fill=color)

    # Calcular tamaño de fuente dinámico
    font_size = max(24, img_size // max(len(text), 1) + 8)
    font_size = min(font_size, img_size // 3)
    font = _find_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = max(0, (img_size - tw) // 2)
    y = max(0, (img_size - th) // 2)

    # Sombra sutil
    shadow_offset = max(2, img_size // 128)
    draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0), font=font)
    # Texto principal
    draw.text((x, y), text, fill=fg_color, font=font)

    return img


# ── Diffusers (modelo de difusión) ──────────────────────────────────────
_diffuser_pipeline = None


def _load_diffuser():
    """Intenta cargar el pipeline de Stable Diffusion."""
    global _diffuser_pipeline
    if _diffuser_pipeline is not None:
        return _diffuser_pipeline

    try:
        from diffusers import StableDiffusionPipeline
        import torch
    except ImportError:
        print("[T2I] diffusers no está instalado. Usando fallback Pillow.")
        return None

    model_id = config.T2I_DIFFUSION_MODEL
    print(f"[T2I] Cargando modelo de difusión: {model_id} ...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        )
        pipe = pipe.to(device)
        _diffuser_pipeline = pipe
        print(f"[T2I] Modelo cargado en {device}.")
        return pipe
    except Exception as e:
        print(f"[T2I] Error cargando modelo de difusión: {e}")
        return None


def _diffusion_render(text: str, seed: int = 42) -> Image.Image:
    """Genera una imagen usando Stable Diffusion."""
    pipe = _load_diffuser()
    if pipe is None:
        raise RuntimeError("No se pudo cargar el modelo de difusión.")

    import torch
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    prompt = (
        f"Artistic elegant calligraphy text '{text}', beautiful typography, "
        f"centered composition, clean background, high quality, sharp focus"
    )

    with torch.no_grad():
        result = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512,
        )
    return result.images[0]


# ── API pública ───────────────────────────────────────────────────────

def generate_image(text: str, seed: int = 42) -> Image.Image:
    """
    Genera una imagen a partir del texto.
    Respeta la variable de entorno T2I_MODE.
    """
    mode = config.T2I_MODE
    if config.T2I_FORCE_FALLBACK:
        mode = "pillow"

    if mode == "diffusion":
        try:
            return _diffusion_render(text, seed=seed)
        except Exception as e:
            print(f"[T2I] Difusión falló ({e}), usando fallback Pillow.")
            return _pillow_render(text, seed=seed)

    return _pillow_render(text, seed=seed)


def generate_and_save(text: str, seed: int = 42) -> str:
    """Genera imagen y la guarda en OUTPUT_DIR. Devuelve la ruta del archivo."""
    img = generate_image(text, seed=seed)
    safe = "".join(c if c.isalnum() else "_" for c in text[:30])
    path = os.path.join(config.OUTPUT_DIR, f"img_{safe}.png")
    img.save(path)
    return path
