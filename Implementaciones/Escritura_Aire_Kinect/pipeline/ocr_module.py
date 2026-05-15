"""
OCR Module — Wrapper de EasyOCR.
Lazy-load para no bloquear el startup de la app.
"""
import os
import cv2
import numpy as np
from PIL import Image
import threading

from . import config

_ocr_reader = None
_ocr_lock = threading.Lock()


def get_ocr_reader():
    """Lazy-load del reader de EasyOCR."""
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader

    with _ocr_lock:
        if _ocr_reader is not None:
            return _ocr_reader

        try:
            import easyocr
            import torch
        except ImportError as e:
            raise RuntimeError(f"EasyOCR no está instalado: {e}")

        use_gpu = False
        if config.OCR_GPU == "auto":
            use_gpu = torch.cuda.is_available()
        elif config.OCR_GPU in ("1", "true", "yes"):
            use_gpu = True

        try:
            _ocr_reader = easyocr.Reader(config.OCR_LANGS, gpu=use_gpu)
        except Exception:
            _ocr_reader = easyocr.Reader(config.OCR_LANGS, gpu=False)

    return _ocr_reader


def ocr_image(image) -> str:
    """
    Recibe una imagen (PIL Image o numpy array) y devuelve el texto reconocido.
    """
    reader = get_ocr_reader()
    if isinstance(image, Image.Image):
        image = np.array(image)
    results = reader.readtext(image, detail=0)
    return " ".join(results).strip()


def ocr_from_opencv(image_bgr) -> str:
    """
    Recibe una imagen BGR de OpenCV, la prepara para OCR y devuelve el texto.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ""
    # Preparar fondo blanco para mejorar OCR
    white_bg = np.ones_like(image_bgr) * 255
    mask = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
    fg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    ocr_img = cv2.add(bg, fg)
    return ocr_image(ocr_img)
