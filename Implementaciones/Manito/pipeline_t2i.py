"""
Pipeline T2I → OCR → Manito

Full flow: Text input → Generate image (T2I model or Pillow fallback) → OCR → Send to hand.
This is the integration point that ties the trained model from Entrenamiento/
with the robotic hand controlled by enviar_letra.py.

Usage standalone:
    python pipeline_t2i.py --text "Santiago"
    python pipeline_t2i.py --file nombres.txt
    python pipeline_t2i.py --text "Duvan" --force-fallback

Usage from enviar_letra.py:
    from pipeline_t2i import Pipeline
    p = Pipeline()
    letters = p.run("Santiago")   # Returns list of recognized letters
"""

import sys
import random
import argparse
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────
MANITO_DIR = Path(__file__).resolve().parent
REPO_ROOT = MANITO_DIR.parent.parent
T2I_DIR = REPO_ROOT / "Entrenamiento" / "T2I_Text2Image"
MODELS_DIR = REPO_ROOT / "Modelos"
OUTPUT_DIR = MANITO_DIR / "pipeline_output"

# Add T2I training code to path so we can import model/dataset
sys.path.insert(0, str(T2I_DIR))

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

# ── Lazy imports from T2I (only when model is available) ──────────────
_model_module = None
_dataset_module = None

def _load_t2i_modules():
    global _model_module, _dataset_module
    if _model_module is None:
        try:
            import model as m
            import dataset as d
            _model_module = m
            _dataset_module = d
        except ImportError as e:
            print(f"[Pipeline] Could not import T2I modules: {e}")
    return _model_module, _dataset_module


# ══════════════════════════════════════════════════════════════════════
# Fallback Renderer (hidden, Pillow-based, offline)
# ══════════════════════════════════════════════════════════════════════

_PALETTES = [
    ((255, 255, 255), (20, 20, 20)),
    ((245, 245, 245), (50, 50, 70)),
    ((0, 0, 0), (240, 240, 240)),
    ((30, 30, 60), (220, 220, 255)),
    ((50, 50, 80), (255, 215, 0)),
    ((250, 248, 240), (60, 60, 80)),
]

def _find_font(size):
    for p in [
        T2I_DIR / "fonts" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if p.exists():
            try: return ImageFont.truetype(str(p), size)
            except (OSError, IOError): continue
    return ImageFont.load_default()

def _fallback_render(text, img_size=64, seed=42):
    """Deterministic Pillow rendering — always works, no GPU needed."""
    random.seed(seed)
    bg, fg = random.choice(_PALETTES)
    img = Image.new("RGB", (img_size, img_size), color=bg)
    draw = ImageDraw.Draw(img)
    font_size = max(12, img_size // max(len(text), 1) + 4)
    font_size = min(font_size, img_size // 2)
    font = _find_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((max(0, (img_size - tw) // 2), max(0, (img_size - th) // 2)),
              text, fill=fg, font=font)
    return img


# ══════════════════════════════════════════════════════════════════════
# OCR wrapper
# ══════════════════════════════════════════════════════════════════════

_ocr_reader = None

def _get_ocr():
    """Lazy-load EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(["es", "en"], gpu=torch.cuda.is_available())
        except ImportError:
            print("[Pipeline] easyocr not installed. pip install easyocr")
            return None
    return _ocr_reader

def ocr_image(image: Image.Image) -> str:
    """Run OCR on a PIL image and return recognized text."""
    reader = _get_ocr()
    if reader is None:
        return ""
    import numpy as np
    img_array = np.array(image)
    results = reader.readtext(img_array, detail=0)
    return " ".join(results).strip()


# ══════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Text → Image (T2I or fallback) → OCR → Letters for hand.

    Usage:
        p = Pipeline()
        letters = p.run("Santiago")    # ['S','A','N','T','I','A','G','O']
        img = p.generate_image("Hola") # PIL Image
    """
    def __init__(self, force_fallback=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._use_fallback = force_fallback

        if not force_fallback:
            self._try_load_model()

    def _try_load_model(self):
        """Try to load the trained T2I model from Modelos/ or checkpoints."""
        model_mod, dataset_mod = _load_t2i_modules()
        if model_mod is None:
            print("[Pipeline] T2I modules not available, using fallback.")
            self._use_fallback = True
            return

        candidates = [
            MODELS_DIR / "t2i_text2image.pth",
            T2I_DIR / "outputs" / "checkpoints" / "best_model.pth",
        ]
        for path in candidates:
            if path.exists():
                try:
                    self.model = model_mod.Text2ImageModel().to(self.device)
                    state = torch.load(path, map_location=self.device, weights_only=True)
                    if "model_state_dict" in state:
                        self.model.load_state_dict(state["model_state_dict"])
                    else:
                        self.model.load_state_dict(state)
                    self.model.eval()
                    print(f"[Pipeline] Model loaded: {path}")
                    return
                except Exception as e:
                    print(f"[Pipeline] Failed to load {path}: {e}")

        print("[Pipeline] No model found, using fallback renderer.")
        self._use_fallback = True

    def generate_image(self, text: str, seed: int = 42) -> Image.Image:
        """Generate image from text using model or fallback."""
        if self._use_fallback or self.model is None:
            return _fallback_render(text, seed=seed)

        try:
            _, dataset_mod = _load_t2i_modules()
            tokens = torch.tensor(
                [dataset_mod.tokenize_text(text)], dtype=torch.long, device=self.device)
            torch.manual_seed(seed)
            noise = torch.randn(1, self.model.noise_dim, device=self.device)

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        img_tensor = self.model(tokens, noise)
                else:
                    img_tensor = self.model(tokens, noise)

            img_tensor = img_tensor.squeeze(0).cpu()
            img_tensor = ((img_tensor + 1) / 2).clamp(0, 1)
            return transforms.ToPILImage()(img_tensor)

        except Exception as e:
            print(f"[Pipeline] Model inference failed: {e}, using fallback.")
            return _fallback_render(text, seed=seed)

    def run(self, text: str, seed: int = 42, save_images: bool = True) -> list[str]:
        """
        Full pipeline: text → image → OCR → letters.

        Args:
            text: Input text to process.
            seed: Random seed for image generation.
            save_images: Save generated and OCR images to pipeline_output/.

        Returns:
            List of recognized letter strings.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate image
        print(f"\n[Pipeline] Input: '{text}'")
        image = self.generate_image(text, seed=seed)
        print(f"[Pipeline] Image generated ({image.size[0]}x{image.size[1]})")

        if save_images:
            safe = "".join(c if c.isalnum() else "_" for c in text[:20])
            img_path = OUTPUT_DIR / f"generated_{safe}.png"
            image.save(img_path)
            print(f"[Pipeline] Saved: {img_path}")

        # Step 2: OCR
        recognized = ocr_image(image)
        print(f"[Pipeline] OCR result: '{recognized}'")

        # Step 3: Extract valid LSC letters
        valid_letters = set("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
        letters = [ch.upper() for ch in recognized if ch.upper() in valid_letters]

        # If OCR failed, fall back to input text directly
        if not letters:
            print("[Pipeline] OCR returned empty, using input text directly.")
            letters = [ch.upper() for ch in text if ch.upper() in valid_letters]

        print(f"[Pipeline] Letters for hand: {letters}")
        return letters

    def run_from_file(self, file_path: str, seed: int = 42) -> list[tuple[str, list[str]]]:
        """Process each line of a .txt file through the pipeline."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    letters = self.run(line, seed=seed)
                    results.append((line, letters))
        return results


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T2I → OCR → Manito Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text to process.")
    group.add_argument("--file", type=str, help="Path to .txt file.")
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pipeline = Pipeline(force_fallback=args.force_fallback)

    if args.text:
        letters = pipeline.run(args.text, seed=args.seed)
        print(f"\nResult: {' → '.join(letters)}")
    elif args.file:
        results = pipeline.run_from_file(args.file, seed=args.seed)
        for text, letters in results:
            print(f"  '{text}' → {letters}")


if __name__ == "__main__":
    main()
