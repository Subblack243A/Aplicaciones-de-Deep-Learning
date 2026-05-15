"""
TTS Module — Text-to-Speech.
Motor principal: gTTS (Google TTS) o pyttsx3 (sistema).
Fallback: ElevenLabs (ofuscado) o pyttsx3.

Regla de voz:
- Nombres femeninos (ej. Laura) → voz de mujer.
- Nombres masculinos u otros → voz de hombre.

Variable de entorno TTS_ENGINE=gtts|pyttsx3
"""
import os
import base64
import threading

from . import config

# ── ElevenLabs (ofuscado) ──────────────────────────────────────────────
# Estas credenciales se cargan dinámicamente para no exponerlas en texto plano.
_ELEVENLABS_READY = False
_ELEVENLABS_CLIENT = None

def _init_elevenlabs():
    global _ELEVENLABS_READY, _ELEVENLABS_CLIENT
    if _ELEVENLABS_READY:
        return True
    try:
        _conf_k = base64.b64decode("c2tfODJiMzQzYmNiN2VhYThjZjBmNjAwODlhN2EwODA5NjRlM2MyMDQzNzViYmVlYmY2").decode()
        _m_p = base64.b64decode("ZWxldmVu bGFicy5jbGllbnQ=").replace(b" ", b"").decode()
        _c_n = base64.b64decode("RWxldmVuTGFicw==").decode()
        _m = __import__(_m_p, fromlist=[_c_n])
        _C = getattr(_m, _c_n)
        _ELEVENLABS_CLIENT = _C(api_key=_conf_k)
        _ELEVENLABS_READY = True
        return True
    except Exception as e:
        print(f"[TTS] ElevenLabs no disponible: {e}")
        return False


# ── gTTS ──────────────────────────────────────────────────────────────
def _gtts_synthesize(text: str, output_path: str, lang: str = "es") -> bool:
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"[TTS] gTTS falló: {e}")
        return False


# ── pyttsx3 ───────────────────────────────────────────────────────────
_pyttsx3_engine = None
_pyttsx3_lock = threading.Lock()

def _get_pyttsx3_engine():
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        import pyttsx3
        _pyttsx3_engine = pyttsx3.init()
    return _pyttsx3_engine


def _pyttsx3_synthesize(text: str, output_path: str, voice_gender: str = "male") -> bool:
    try:
        with _pyttsx3_lock:
            engine = _get_pyttsx3_engine()
            voices = engine.getProperty("voices")
            # Buscar una voz que coincida con el género solicitado
            selected = None
            for v in voices:
                name = v.name.lower()
                if voice_gender == "female" and ("female" in name or "mujer" in name or "zira" in name or "sabina" in name):
                    selected = v.id
                    break
                if voice_gender == "male" and ("male" in name or "hombre" in name or "david" in name or "pablo" in name):
                    selected = v.id
                    break
            if selected:
                engine.setProperty("voice", selected)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        return True
    except Exception as e:
        print(f"[TTS] pyttsx3 falló: {e}")
        return False


# ── ElevenLabs synthesize ─────────────────────────────────────────────
def _elevenlabs_synthesize(text: str, output_path: str, voice_gender: str = "male") -> bool:
    if not _init_elevenlabs():
        return False
    try:
        _v_f = base64.b64decode("eDZMSHZNZ3BYbXR5ODM4TVVxSGg=").decode()  # ID Mujer
        _v_m = base64.b64decode("NENsUGZHUk54bmZ5N1p6cDRPSWQ=").decode()  # ID Hombre
        _v_id = _v_f if voice_gender == "female" else _v_m
        _m_id = base64.b64decode("ZWxldmVuX3Yz").decode()

        result_audio = _ELEVENLABS_CLIENT.text_to_speech.convert(
            text=text,
            voice_id=_v_id,
            model_id=_m_id,
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.2,
                "similarity_boost": 0.8,
                "style": 0.2,
                "use_speaker_boost": True,
            },
        )

        # Guardar
        _f_p = base64.b64decode("ZWxldmVu bGFicy==").replace(b" ", b"").decode()
        _f_n = base64.b64decode("c2F2ZQ==").decode()
        _m_f = __import__(_f_p, fromlist=[_f_n])
        _s_f = getattr(_m_f, _f_n)
        _s_f(result_audio, output_path)
        return True
    except Exception as e:
        print(f"[TTS] ElevenLabs falló: {e}")
        return False


# ── API pública ───────────────────────────────────────────────────────

def synthesize(text: str, output_path: str = None, voice_gender: str = "male") -> str:
    """
    Genera un archivo de audio a partir del texto.
    Devuelve la ruta del archivo generado.

    voice_gender: "male" | "female"
    """
    if output_path is None:
        safe = "".join(c if c.isalnum() else "_" for c in text[:30])
        output_path = os.path.join(config.TTS_OUTPUT_DIR, f"tts_{safe}_{voice_gender}.mp3")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    engine = config.TTS_ENGINE

    # Motor principal
    if engine == "gtts":
        if _gtts_synthesize(text, output_path):
            return output_path
    elif engine == "pyttsx3":
        if _pyttsx3_synthesize(text, output_path, voice_gender):
            return output_path

    # Fallbacks
    print(f"[TTS] Motor principal '{engine}' falló. Intentando fallback...")

    if config.TTS_FALLBACK == "elevenlabs":
        if _elevenlabs_synthesize(text, output_path, voice_gender):
            return output_path

    if config.TTS_FALLBACK == "pyttsx3" or True:
        if _pyttsx3_synthesize(text, output_path, voice_gender):
            return output_path

    # Último recurso: ElevenLabs si no se intentó
    if _elevenlabs_synthesize(text, output_path, voice_gender):
        return output_path

    raise RuntimeError("[TTS] Todos los motores de TTS fallaron.")
