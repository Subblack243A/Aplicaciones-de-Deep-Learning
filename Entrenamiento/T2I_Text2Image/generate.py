"""
T2I Text-to-Image — Inferencia / Generación de imágenes.

Carga el mejor checkpoint entrenado y genera una imagen PNG
a partir de cualquier texto que le pases.

Uso básico:
    python generate.py --text "Santiago"
    python generate.py --text "Hola Mundo"
    python generate.py --text "Deep Learning"

Opciones avanzadas:
    # Generar varias palabras en un solo comando (una imagen por texto)
    python generate.py --text "Duvan" "Carlos" "Sofia"

    # Especificar ruta del checkpoint manualmente
    python generate.py --text "Luna" --checkpoint outputs/checkpoints/best_model.pth

    # Especificar carpeta de salida
    python generate.py --text "Robot" --out-dir mis_imagenes

    # Mostrar grilla de todos los textos generados en un solo PNG
    python generate.py --text "Sol" "Luna" "Pixel" --grid
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Rutas por defecto (misma estructura que train.py) ─────────────────
BASE_DIR       = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
DEFAULT_CKPT   = CHECKPOINT_DIR / "best_model.pth"
DEFAULT_OUT    = BASE_DIR / "outputs" / "generated"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path) -> "Text2ImageModel":
    """
    Carga Text2ImageModel desde un checkpoint .pth.
    Acepta tanto el formato completo {model_state_dict, epoch, ...}
    que guarda train.py, como un state_dict directo.
    """
    # Import local para no depender del PYTHONPATH del usuario
    try:
        from model import Text2ImageModel
    except ModuleNotFoundError:
        print("[ERROR] No se encontró model.py. Asegúrate de ejecutar este script")
        print("        desde la misma carpeta donde está model.py.")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"[ERROR] No se encontró el checkpoint en: {checkpoint_path}")
        print("        Entrena el modelo primero con:  python train.py")
        sys.exit(1)

    raw = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

    # Soporte para ambos formatos de guardado
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        epoch_info = raw.get("epoch", "?")
        print(f"  Checkpoint completo cargado (guardado en época {epoch_info + 1})")
    else:
        state_dict = raw
        print("  State dict directo cargado")

    model = Text2ImageModel()
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def text_to_tokens(text: str) -> torch.Tensor:
    """
    Convierte un string en tensor de tokens (igual que dataset.py).
    Importa tokenize_text si está disponible; si no, lo reimplementa.
    """
    try:
        from dataset import tokenize_text
        tokens = tokenize_text(text)
    except ModuleNotFoundError:
        # Fallback inline por si dataset.py no está en el path
        from model import MAX_TEXT_LEN, VOCAB_SIZE
        tokens = [min(max(ord(c), 1), VOCAB_SIZE - 1) for c in text[:MAX_TEXT_LEN]]
        tokens += [0] * (MAX_TEXT_LEN - len(tokens))

    return torch.tensor([tokens], dtype=torch.long, device=DEVICE)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convierte el tensor de salida del modelo (B, 3, H, W) en rango [-1, 1]
    a una imagen PIL lista para guardar.
    """
    img = tensor.squeeze(0).cpu().float()   # (3, H, W)
    img = (img * 0.5 + 0.5).clamp(0, 1)    # [-1,1] → [0,1]
    return TF.to_pil_image(img)


def generate_one(model, text: str) -> Image.Image:
    """Genera una sola imagen PIL a partir de un texto."""
    tokens = text_to_tokens(text)
    with torch.no_grad():
        output = model.generate(tokens, device=DEVICE)
    return tensor_to_pil(output)


def save_image(img: Image.Image, text: str, out_dir: Path) -> Path:
    """Guarda la imagen con un nombre derivado del texto."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in text[:40]).strip()
    safe_name = safe_name.replace(" ", "_")
    out_path  = out_dir / f"{safe_name}.png"
    img.save(out_path)
    return out_path


def save_grid(images: list[Image.Image], texts: list[str], out_dir: Path) -> Path:
    """
    Guarda todas las imágenes generadas en un solo PNG tipo grilla,
    con el texto correspondiente debajo de cada imagen.
    """
    n    = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2.5, rows * 3.0))
    gs  = gridspec.GridSpec(rows, cols, figure=fig,
                             hspace=0.45, wspace=0.15)

    for i, (img, text) in enumerate(zip(images, texts)):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(text, fontsize=9, fontweight="bold",
                     pad=4, wrap=True)
        ax.axis("off")

    # Ocultar subplots vacíos si n no es múltiplo de cols
    for j in range(n, rows * cols):
        row, col = divmod(j, cols)
        fig.add_subplot(gs[row, col]).axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    return grid_path


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera imágenes desde texto usando el modelo T2I entrenado.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--text", nargs="+", required=True,
        metavar="TEXTO",
        help="Uno o varios textos a renderizar. Ejemplo: --text \"Duvan\" \"Sofia\"",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        metavar="RUTA",
        help=f"Ruta al archivo .pth del checkpoint.\n"
             f"Por defecto: {DEFAULT_CKPT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        metavar="CARPETA",
        help=f"Carpeta donde se guardan las imágenes generadas.\n"
             f"Por defecto: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Además de los PNGs individuales, guarda una grilla con todos los resultados.",
    )
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  T2I Inferencia   |   device: {DEVICE}")
    print(f"{'='*55}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Salida     : {args.out_dir}")
    print(f"  Textos     : {args.text}\n")

    # 1. Cargar modelo
    print("Cargando modelo...")
    model = load_model(args.checkpoint)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros totales: {params:,}\n")

    # 2. Generar imágenes
    images   = []
    saved    = []

    for text in args.text:
        print(f"  Generando: \"{text}\"  ...", end=" ", flush=True)
        img      = generate_one(model, text)
        out_path = save_image(img, text, args.out_dir)
        images.append(img)
        saved.append(out_path)
        print(f"→ {out_path}")

    # 3. Grilla opcional
    if args.grid and len(images) > 1:
        grid_path = save_grid(images, args.text, args.out_dir)
        print(f"\n  Grilla guardada → {grid_path}")

    print(f"\n{'='*55}")
    print(f"  ✓ {len(images)} imagen(es) guardada(s) en {args.out_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()