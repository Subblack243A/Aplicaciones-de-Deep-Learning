"""
T2I Dataset — Synthetic generator + DataLoader in one file.

generate_dataset() : Creates text→image pairs using Pillow rendering
TextImageDataset   : PyTorch Dataset that loads from manifest.json
create_dataloaders(): Returns (train_loader, val_loader)
tokenize_text()    : Char→ASCII token ID conversion
"""

import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms

from model import IMG_SIZE, MAX_TEXT_LEN, VOCAB_SIZE

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
FONTS_DIR = Path(__file__).resolve().parent / "fonts"
NUM_SAMPLES = 10000
BATCH_SIZE = 32
VAL_SPLIT = 0.15

SAMPLE_NAMES = [
    "Duvan", "Santiago", "David", "Oscar", "Felipe",
    "Laura", "Estefania", "Carlos", "Juan", "Maria",
    "Andres", "Camila", "Nicolas", "Valentina", "Diego",
    "Isabella", "Sebastian", "Daniela", "Alejandro", "Sofia",
    "Miguel", "Natalia", "Jorge", "Carolina", "Ricardo",
    "Ana", "Pedro", "Lucia", "Fernando", "Gabriela",
    "Mateo", "Mariana", "Luis", "Juliana", "Daniel",
    "Paula", "Cristian", "Manuela", "Sergio", "Tatiana",
    "Hola", "Mundo", "Python", "Deep", "Learning",
    "Neural", "Robot", "Mano", "Texto", "Imagen",
    "Azul", "Rojo", "Verde", "Luz", "Noche",
    "Sol", "Luna", "Pixel", "Color", "Letra",
]

COLOR_PALETTES = [
    ((255, 255, 255), (20, 20, 20)),
    ((245, 245, 245), (50, 50, 70)),
    ((0, 0, 0), (240, 240, 240)),
    ((30, 30, 60), (220, 220, 255)),
    ((60, 30, 30), (255, 200, 200)),
    ((30, 60, 30), (200, 255, 200)),
    ((240, 240, 220), (40, 40, 60)),
    ((50, 50, 80), (255, 215, 0)),
    ((250, 248, 240), (60, 60, 80)),
    ((35, 35, 55), (180, 230, 255)),
    ((25, 50, 50), (150, 255, 200)),
]


# ── Tokenization ──────────────────────────────────────────────────────

def tokenize_text(text: str, max_len: int = MAX_TEXT_LEN) -> list[int]:
    tokens = [min(max(ord(c), 1), VOCAB_SIZE - 1) for c in text[:max_len]]
    return tokens + [0] * (max_len - len(tokens))

def detokenize(token_ids) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return "".join(chr(t) for t in token_ids if t > 0)


# ── Font helper ───────────────────────────────────────────────────────

def _find_font(size: int):
    for p in [
        FONTS_DIR / "DejaVuSans.ttf",
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


# ── Render single image ──────────────────────────────────────────────

def render_text_image(text, img_size=IMG_SIZE, font_size=None, bg_color=None, text_color=None):
    """
    Renderiza el texto de forma DETERMINISTA.
 
    Se crea un generador de números aleatorios LOCAL (random.Random) cuya semilla
    se deriva del hash MD5 del texto. Esto garantiza que:
      - El mismo texto → siempre los mismos colores y tamaño de fuente.
      - El RNG global (usado para shuffle del dataset) no se ve afectado.
      - Si el dataset se regenera, las imágenes son idénticas a las anteriores.
 
    Los parámetros font_size, bg_color y text_color siguen siendo sobreescribibles
    desde fuera (útil para tests o visualizaciones manuales).
    """
    text_seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(text_seed)
 
    if font_size is None:
        font_size = rng.randint(14, 28)
 
    if bg_color is None or text_color is None:
        bg_gray_value = rng.randint(0, 255)
 
        if bg_gray_value > 127:
            text_gray_value = rng.randint(0, 80)
        else:
            text_gray_value = rng.randint(175, 255)
 
        bg_color   = (bg_gray_value,   bg_gray_value,   bg_gray_value)
        text_color = (text_gray_value, text_gray_value, text_gray_value)
 
    # Crear imagen
    img  = Image.new("RGB", (img_size, img_size), color=bg_color)
    draw = ImageDraw.Draw(img)
 
    margin     = 10
    max_width  = img_size - margin
    max_height = img_size - margin
 
    font = _find_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
 
    # Autoescalado: reducir fuente hasta que el texto quepa
    while (tw > max_width or th > max_height) and font_size > 8:
        font_size -= 1
        font = _find_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
 
    x = max(0, (img_size - tw) // 2)
    y = max(0, (img_size - th) // 2)
 
    draw.text((x, y), text, fill=text_color, font=font)
    return img

# ── Generate full dataset ─────────────────────────────────────────────

def generate_dataset(output_dir=None, num_samples=NUM_SAMPLES, text_source_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    if text_source_dir and Path(text_source_dir).exists():
        for txt_file in sorted(Path(text_source_dir).glob("*.txt")):
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) <= MAX_TEXT_LEN:
                        texts.append(line)

    while len(texts) < num_samples:
        name = random.choice(SAMPLE_NAMES)
        texts.append(random.choice([name, name.upper(), name.lower()]))

    random.shuffle(texts)
    texts = texts[:num_samples]

    manifest = []
    print(f"Generating {len(texts)} synthetic text→image pairs...")
    for idx, text in enumerate(texts):
        img = render_text_image(text)
        fname = f"sample_{idx:05d}.png"
        img.save(images_dir / fname)
        manifest.append({"id": idx, "text": text, "image": fname})
        if (idx + 1) % 1000 == 0:
            print(f"  [{idx + 1}/{len(texts)}] generated...")

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Dataset ready: {len(manifest)} samples → {output_dir}")
    return output_dir


# ── PyTorch Dataset ───────────────────────────────────────────────────

class TextImageDataset(Dataset):
    def __init__(self, data_dir=None, transform=None):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.images_dir = self.data_dir / "images"
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest at {manifest_path}. Run: python train.py --generate-data-only")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        image = Image.open(self.images_dir / entry["image"]).convert("RGB")
        image = self.transform(image)
        tokens = torch.tensor(tokenize_text(entry["text"]), dtype=torch.long)
        return tokens, image


# ── DataLoader factory ────────────────────────────────────────────────

def create_dataloaders(data_dir=None, batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    dataset = TextImageDataset(data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"DataLoaders: {train_size} train / {val_size} val (batch={batch_size})")
    return train_loader, val_loader


if __name__ == "__main__":
    generate_dataset()
