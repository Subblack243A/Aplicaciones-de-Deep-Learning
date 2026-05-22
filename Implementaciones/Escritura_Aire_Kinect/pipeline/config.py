"""
Configuración global del pipeline.
Usa variables de entorno para permitir switches sin modificar código.
"""
import os

# ── Directorios ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── T2I (Text-to-Image) ───────────────────────────────────────────────
# Modos: "diffusion" | "pillow"
T2I_MODE = os.getenv("T2I_MODE", "pillow")
# Modelo de diffusión a usar (HuggingFace model ID)
T2I_DIFFUSION_MODEL = os.getenv("T2I_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5")
# Si True, fuerza fallback incluso si el modelo existe
T2I_FORCE_FALLBACK = os.getenv("T2I_FORCE_FALLBACK", "false").lower() in ("1", "true", "yes")

# ── TTS (Text-to-Speech) ──────────────────────────────────────────────
# Motor principal: "gtts" | "pyttsx3"
TTS_ENGINE = os.getenv("TTS_ENGINE", "gtts")
# Fallback: "elevenlabs" | "pyttsx3" | None
TTS_FALLBACK = os.getenv("TTS_FALLBACK", "elevenlabs")
# Directorio para audios generados
TTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "audio")
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

# ── OCR ───────────────────────────────────────────────────────────────
OCR_LANGS = os.getenv("OCR_LANGS", "es,en").split(",")
OCR_GPU = os.getenv("OCR_GPU", "auto")

# ── Hand (Mano Robótica) ──────────────────────────────────────────────
# Modos: "websocket" | "mock" (simulado, no requiere ESP32)
HAND_MODE = os.getenv("HAND_MODE", "websocket")
HAND_IP = os.getenv("HAND_IP", "10.157.97.197")
HAND_PORT = int(os.getenv("HAND_PORT", "81"))

# ── Pipeline ──────────────────────────────────────────────────────────
# Delay entre letras enviadas a la mano (ms)
HAND_DELAY_BETWEEN_LETTERS_MS = int(os.getenv("HAND_DELAY_MS", "1500"))
