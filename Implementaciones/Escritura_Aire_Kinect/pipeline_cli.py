"""
pipeline_cli.py — Script standalone para ejecutar el pipeline completo sin Kinect.

Uso:
    python pipeline_cli.py --text "Laura"
    python pipeline_cli.py --text "Santiago" --output-dir ./resultados
    T2I_MODE=diffusion python pipeline_cli.py --text "Maria"
    HAND_MODE=websocket HAND_IP=manito.local python pipeline_cli.py --text "Hola"

Flujo:
    1. Recibe texto desde línea de comandos.
    2. Genera imagen (T2I).
    3. Genera audio (TTS) con detección de género.
    4. Simula/Envía a la mano robótica (LSC).
    5. Guarda todo en el directorio de salida.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Asegurar que el directorio del script esté en el path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline.t2i_module import generate_and_save
from pipeline.tts_module import synthesize
from pipeline.gender_detector import detect_gender
from pipeline.hand_module import HandController
from pipeline import config as pipeline_config


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Full — Texto → Imagen + Audio + Mano (sin Kinect)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Texto a procesar (ej. 'Laura', 'Santiago')."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directorio de salida (default: pipeline_output del config)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para la generación de imagen (default: 42)."
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Reproduce el audio automáticamente tras generarlo (requiere pygame)."
    )
    parser.add_argument(
        "--send-hand",
        action="store_true",
        help="Envía el texto a la mano robótica (requiere HAND_MODE=websocket)."
    )

    args = parser.parse_args()
    texto = args.text.strip()

    if not texto:
        print("❌ Error: el texto no puede estar vacío.")
        sys.exit(1)

    # Directorio de salida
    output_dir = args.output_dir or pipeline_config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Pipeline Full — Input: '{texto}'")
    print(f"📁 Output dir: {output_dir}")
    print("=" * 60)
    print()

    # ── 1. T2I: Generar imagen ──
    print("[1/4] 🎨 Generando imagen (T2I)...")
    img_path = None
    try:
        img_path = generate_and_save(texto, seed=args.seed)
        print(f"      ✅ Imagen guardada: {img_path}")
    except Exception as e:
        print(f"      ❌ Error T2I: {e}")

    # ── 2. TTS: Generar audio ──
    print("[2/4] 🔊 Generando audio (TTS)...")
    audio_path = None
    try:
        gender = detect_gender(texto)
        gender_label = "mujer" if gender == "female" else "hombre"
        print(f"      🧠 Género detectado: {gender_label}")

        safe_name = "".join(c if c.isalnum() else "_" for c in texto[:30])
        audio_path = os.path.join(output_dir, f"{safe_name}_{gender}.mp3")
        audio_path = synthesize(texto, output_path=audio_path, voice_gender=gender)
        print(f"      ✅ Audio guardado: {audio_path}")
    except Exception as e:
        print(f"      ❌ Error TTS: {e}")

    # ── 3. Reproducir audio (opcional) ──
    if args.play_audio and audio_path and os.path.exists(audio_path):
        print("[3/4] ▶️  Reproduciendo audio...")
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            # Esperar a que termine la reproducción
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("      ✅ Reproducción completada.")
        except Exception as e:
            print(f"      ❌ Error reproduciendo audio: {e}")
    else:
        print("[3/4] ⏭️  Reproducción de audio omitida.")

    # ── 4. Mano robótica ──
    print("[4/4] ✋ Enviando a la mano robótica...")
    hand = HandController()
    try:
        if args.send_hand:
            if hand.connect():
                hand.send_text(texto)
                # Esperar a que termine la cola
                while hand._queue_active:
                    time.sleep(0.1)
                hand.disconnect()
                print("      ✅ Mano: envío completado.")
            else:
                print("      ⚠️  No se pudo conectar a la mano.")
        else:
            # Modo mock: solo mostrar qué haría
            print(f"      🧪 [MOCK] La mano deletrearía: {' '.join(texto.upper())}")
            print("      💡 Usa --send-hand para enviar realmente (requiere HAND_MODE=websocket).")
    except Exception as e:
        print(f"      ❌ Error con la mano: {e}")

    # ── Resumen ──
    print()
    print("=" * 60)
    print("📊 RESUMEN DEL PIPELINE")
    print("=" * 60)
    print(f"   📝 Texto:        {texto}")
    print(f"   🖼  Imagen:       {img_path or '❌ Falló'}")
    print(f"   🔊 Audio:        {audio_path or '❌ Falló'}")
    print(f"   ✋ Mano:         {'Enviado' if args.send_hand else 'Mock/Simulado'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
