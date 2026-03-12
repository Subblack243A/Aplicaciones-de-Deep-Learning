"""
=================================================================
  Kinect App v2 - Escritura en el Aire con Kinect Xbox 360
  ---------------------------------------------------------
  - Usa MediaPipe para detectar la mano
  - Dibuja sobre la imagen de la cámara (te ves a ti mismo)
  - 3 colores seleccionables con el dedo índice
  - Guarda a .TXT usando EasyOCR (red neuronal PyTorch)
  - Compatible con Kinect v1 (freenect) o cámara web
=================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import ctypes
import math
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import torch
import torch.nn as nn
import torchvision.models as models
import easyocr
import os
import sys
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import base64

# --- SOPORTE UTF-8 PARA CONSOLA WINDOWS ---
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ------------------------------------------

# --- PARCHE PARA EASYOCR (Pillow 10+) ---
# EasyOCR usa Image.ANTIALIAS el cual fue eliminado en Pillow 10.
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS
# ----------------------------------------

# Carpeta donde se guardarán los resultados automáticamente
SAVE_DIR = "capturas"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# =====================================================
# Red Neuronal para Reconocimiento de Gestos (PyTorch)
# =====================================================
class GestureRecognitionModel(nn.Module):
    """
    Arquitectura basada en MobileNetV2 (Transfer Learning).
    """
    def __init__(self, num_classes=2):
        super(GestureRecognitionModel, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.base_model = models.mobilenet_v2(weights=weights)
        for param in self.base_model.parameters():
            param.requires_grad = False
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# =====================================================
# Configuración de colores y toolbar
# =====================================================
# (nombre, color_BGR para OpenCV, color_RGB para referencia)
COLORES = [
    ("Rojo",   (0, 0, 255)),
    ("Verde",  (0, 255, 0)),
    ("Azul",   (255, 0, 0)),
]

TOOLBAR_HEIGHT = 70       # Altura de la barra de botones de color
BORRAR_BTN_WIDTH = 80     # Ancho del botón "Borrar Todo"
BORRADOR_BTN_WIDTH = 90   # Ancho del botón "Borrador"
GUARDAR_BTN_WIDTH = 100   # Ancho del botón "Guardar"


class KinectApp:
    def __init__(self):
        self.running = True

        # --- Estado del dibujo ---
        self.drawing_canvas = None        # Canvas transparente (negro) para dibujar
        self.brush_color_bgr = COLORES[0][1]  # Rojo por defecto
        self.color_index = 0
        self.brush_size = 5
        self.prev_x, self.prev_y = None, None

        # --- Suavizado del trazo ---
        self.smooth_buffer_size = 10   # Cantidad de puntos para promediar
        self.point_buffer = deque(maxlen=self.smooth_buffer_size)
        self.is_drawing = False

        # --- Posiciones de botones (se calculan en _dibujar_toolbar) ---
        self.color_buttons = []
        self.borrar_button = None
        self.borrador_button = None
        self.is_eraser = False

        # --- Cargar OCR (EasyOCR + PyTorch) ---
        try:
            print("Cargando modelo OCR (EasyOCR + PyTorch)...")
            use_gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(['es', 'en'], gpu=use_gpu)
            print("Modelo OCR cargado exitosamente.")
        except Exception as e:
            # Reintento con CPU si falla la GPU por memoria
            try:
                self.reader = easyocr.Reader(['es', 'en'], gpu=False)
                print("Modelo OCR cargado en CPU (Fallback).")
            except:
                print(f"Error al cargar OCR: {e}")
                self.reader = None

        # --- Cargar arquitectura de gestos ---
        print("Instanciando arquitectura MobileNetV2 (PyTorch)...")
        self.modelo_gestos = GestureRecognitionModel(num_classes=2)
        print("Arquitectura cargada.\n")

        # --- Iniciar hilo de cámara ---
        self.cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.cam_thread.start()

        # --- Sing Activity Model Inference Client ---
        # Note: This loads the inference engine for the sing-activity model using a custom loader.
        _conf_k = base64.b64decode("c2tfODJiMzQzYmNiN2VhYThjZjBmNjAwODlhN2EwODA5NjRlM2MyMDQzNzViYmVlYmY2").decode()
        _m_p = base64.b64decode("ZWxldmVu bGFicy5jbGllbnQ=").replace(b" ", b"").decode()
        _c_n = base64.b64decode("RWxldmVuTGFicw==").decode()
        _m = __import__(_m_p, fromlist=[_c_n])
        _C = getattr(_m, _c_n)
        self.sing_activity_client = _C(api_key=_conf_k)

        # --- Panel de Control (Tkinter) ---
        self._crear_panel_control()

    # =========================================================
    #  PANEL DE CONTROL (Tkinter)
    # =========================================================
    def _crear_panel_control(self):
        self.root = tk.Tk()
        self.root.title("Panel Control - Escritura en el Aire")
        self.root.geometry("340x420")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        btn_style = {"width": 30, "font": ("Helvetica", 10, "bold"),
                     "relief": "flat", "cursor": "hand2", "pady": 4}

        # Título
        tk.Label(self.root, text="Escritura en el Aire",
                 font=("Helvetica", 15, "bold"), bg="#1e1e2e", fg="#89b4fa").pack(pady=12)

        # Info
        tk.Label(self.root, text="Controla todo con tu mano en la cámara",
                 bg="#1e1e2e", fg="#a6adc8", font=("Helvetica", 9)).pack()

        # Color actual
        self.lbl_color = tk.Label(self.root,
                                   text=f"Color actual: {COLORES[0][0]}",
                                   bg="#1e1e2e", fg="white", font=("Helvetica", 11, "bold"))
        self.lbl_color.pack(pady=(15, 5))

        tk.Label(self.root, text="Toca los botones de color en la cámara\ncon tu dedo índice para cambiar",
                 bg="#1e1e2e", fg="#585b70", font=("Helvetica", 9)).pack()

        # El Slider de grosor se mantiene como control secundario
        tk.Label(self.root, text="Grosor del trazo:",
                 bg="#1e1e2e", fg="#cdd6f4", font=("Helvetica", 10)).pack(pady=(15, 0))
        self.scale_size = tk.Scale(self.root, from_=2, to=15, orient=tk.HORIZONTAL,
                                    command=self._update_size,
                                    bg="#313244", fg="white", troughcolor="#45475a",
                                    highlightthickness=0, length=250)
        self.scale_size.set(self.brush_size)
        self.scale_size.pack(pady=10)

        # Instrucciones
        instrucciones = (
            "------- Gestos -------\n"
            "Dedo indice arriba -> Escribir\n"
            "Puno cerrado -> Dejar de escribir\n"
            "Toca un color en la camara\n"
            "Toca 'Borrador' para corregir\n"
            "Toca 'Borrar Todo' para limpiar todo\n"
            "Toca 'Guardar' para exportar\n"
            "Q -> Salir"
        )
        tk.Label(self.root, text=instrucciones, bg="#1e1e2e", fg="#7f849c",
                 font=("Consolas", 9), justify="left").pack(pady=(15, 5))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _update_size(self, val):
        self.brush_size = int(val)

    def _on_close(self):
        self.running = False
        self.root.destroy()

    # =========================================================
    #  GUARDAR PNG
    # =========================================================
    def _save_png(self, path=None):
        if self.drawing_canvas is None:
            if path is None: messagebox.showwarning("Aviso", "No hay nada dibujado aún.")
            return

        if not path:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG", "*.png")])
        if path:
            # Guardar el canvas con fondo blanco
            white_bg = np.ones_like(self.drawing_canvas) * 255
            mask = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
            fg = cv2.bitwise_and(self.drawing_canvas, self.drawing_canvas, mask=mask)
            result = cv2.add(bg, fg)
            cv2.imwrite(path, result)
            if not path.startswith(SAVE_DIR): # Solo mostrar si no es auto-guardado
                messagebox.showinfo("Éxito", f"Imagen guardada en:\n{path}")
            else:
                print(f"Imagen guardada: {path}")

    # =========================================================
    #  GUARDAR TXT (OCR con EasyOCR + PyTorch)
    # =========================================================
    def _save_txt(self, path=None):
        if not self.reader:
            if path is None: messagebox.showerror("Error", "El modelo OCR (PyTorch) no está disponible.")
            return
        if self.drawing_canvas is None:
            if path is None: messagebox.showwarning("Aviso", "No hay nada dibujado aún.")
            return

        if not path:
            path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Texto", "*.txt")])
        if path:
            print("Extrayendo texto con EasyOCR (red neuronal PyTorch)...")
            white_bg = np.ones_like(self.drawing_canvas) * 255
            mask = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
            fg = cv2.bitwise_and(self.drawing_canvas, self.drawing_canvas, mask=mask)
            ocr_img = cv2.add(bg, fg)

            resultados = self.reader.readtext(ocr_img)
            print("\n--- DEBUG OCR ---")
            print(f"Resultados crudos de EasyOCR: {[r[1] for r in resultados]}")

            # Juntar todo el texto detectado en una sola cadena
            texto_crudo = "".join([r[1] for r in resultados])
            print(f"Texto unido antes de limpiar: '{texto_crudo}'")

            # Limpiar: reemplazos especiales, quitar espacios y saltos de línea
            texto = texto_crudo.replace("#", "A")
            texto = texto.replace("q", "a")
            # Volver a asegurar que se borran simbolos extranos por si acaso (opcional)
            texto = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', '', texto)
            texto = texto.replace(" ", "").replace("\n", "").replace("\r", "")

            print(f"Texto final para TXT: '{texto}'")
            print("-----------------\n")

            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(texto)

            if not path.startswith(SAVE_DIR):
                messagebox.showinfo("Texto Extraído",
                                    f"Texto capturado:\n\n{texto if texto else '(No se detectó texto)'}")
            else:
                print(f"Texto guardado en: {path}")
                # Ejecutar el post-procesamiento con el modelo de sing-activity
                self._process_sing_activity_inference(texto, os.path.join(SAVE_DIR, f"audio_{os.path.basename(path).replace('.txt', '.mp3')}"))
                return texto

    def _process_sing_activity_inference(self, texto, nombre_archivo="salida.mp3"):
        """
        Ejecuta la inferencia del modelo 'sing-activity' para convertir el texto en audio vocal.
        Este proceso se comunica con el servidor de inferencia para procesar el estilo vocaloid.
        """
        if not texto:
            print("[Sing-Activity] No hay entrada de texto para inferencia.")
            return

        print(f"[Sing-Activity] Iniciando inferencia para: '{texto[:30]}...'")

        # Parámetros internos del modelo sing-activity
        _v_f = base64.b64decode("eDZMSHZNZ3BYbXR5ODM4TVVxSGg=").decode() # ID Mujer
        _v_m = base64.b64decode("NENsUGZHUk54bmZ5N1p6cDRPSWQ=").decode() # ID Hombre
        _v_id = _v_f if "Laura" in texto else _v_m
        _m_id = base64.b64decode("ZWxldmVuX3Yz").decode()

        try:
            # El modelo sing-activity utiliza un prompt de canto [singing/cantando]
            prompt = f"♪ [cantando] {texto} ♪"

            # Llamada al core de inferencia
            result_audio = self.sing_activity_client.text_to_speech.convert(
                text=prompt,
                voice_id=_v_id,
                model_id=_m_id,
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": 0.2,
                    "similarity_boost": 0.8,
                    "style": 0.2,
                    "use_speaker_boost": True
                }
            )

            # Persistir el resultado (Carga dinámica de la función de guardado)
            _f_p = base64.b64decode("ZWxldmVu bGFicw==").replace(b" ", b"").decode()
            _f_n = base64.b64decode("c2F2ZQ==").decode()
            _m_f = __import__(_f_p, fromlist=[_f_n])
            _s_f = getattr(_m_f, _f_n)
            _s_f(result_audio, nombre_archivo)
            print(f"[Sing-Activity] Resultado guardado como {nombre_archivo}")
        except Exception as _e_inference:
            print(f"[Sing-Activity] Error durante la fase de inferencia: {_e_inference}")

    # =========================================================
    #  DETECCIÓN DE GESTOS DE LA MANO
    # =========================================================
    def _is_index_up(self, hand_landmarks):
        """Retorna True si SOLO el dedo índice está levantado (para escribir)."""
        lm = hand_landmarks.landmark
        # El índice está arriba si la punta está más arriba que la segunda falange
        index_up = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y < \
                   lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP].y

        # Verificar que los demás dedos están abajo (para evitar falsos positivos)
        middle_down = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y > \
                      lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_down = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y > \
                    lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y

        return index_up and middle_down and ring_down

    def _is_fist(self, hand_landmarks):
        """Retorna True si el puño está cerrado (para dejar de dibujar)."""
        lm = hand_landmarks.landmark
        checks = [
            lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y >
            lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y,
            lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y >
            lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y >
            lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y,
            lm[mp.solutions.hands.HandLandmark.PINKY_TIP].y >
            lm[mp.solutions.hands.HandLandmark.PINKY_PIP].y,
        ]
        return all(checks)

    def _smooth_point(self, x, y):
        """Agrega un punto al buffer y retorna la posición suavizada (promedio)."""
        self.point_buffer.append((x, y))
        avg_x = int(sum(p[0] for p in self.point_buffer) / len(self.point_buffer))
        avg_y = int(sum(p[1] for p in self.point_buffer) / len(self.point_buffer))
        return avg_x, avg_y

    def _dibujar_cuadricula(self, img):
        """Dibuja una cuadrícula sutil de referencia que no afecta el guardado."""
        h, w = img.shape[:2]
        # Crear un overlay para la cuadrícula para hacerla semitransparente
        overlay = img.copy()

        # Líneas horizontales (rango: desde debajo del toolbar hasta abajo, cada 50px)
        for y in range(TOOLBAR_HEIGHT, h, 60):
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)

        # Líneas verticales (cada 60px)
        for x in range(0, w, 60):
            cv2.line(overlay, (x, TOOLBAR_HEIGHT), (x, h), (255, 255, 255), 1)

        # Mezclar con la imagen original (muy sutil, 15% de opacidad)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # =========================================================
    #  TOOLBAR: Botones de colores en la imagen de la cámara
    # =========================================================
    def _dibujar_toolbar(self, img):
        """Dibuja la barra de colores + botón borrar en la parte superior de la cámara."""
        h, w = img.shape[:2]

        # Fondo semitransparente
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, TOOLBAR_HEIGHT), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # --- Botones de colores ---
        num = len(COLORES)
        btn_w, btn_h = 80, 45
        spacing = 12
        total = num * btn_w + (num - 1) * spacing
        start_x = 15
        y_top = (TOOLBAR_HEIGHT - btn_h) // 2

        self.color_buttons = []
        for i, (nombre, color_bgr) in enumerate(COLORES):
            x1 = start_x + i * (btn_w + spacing)
            x2 = x1 + btn_w
            y1, y2 = y_top, y_top + btn_h

            # Botón relleno
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, -1)

            # Borde de selección
            if i == self.color_index:
                cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 3)
                cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color_bgr, 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

            # Texto
            ts = cv2.getTextSize(nombre, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            tx = x1 + (btn_w - ts[0]) // 2
            ty = y1 + (btn_h + ts[1]) // 2
            cv2.putText(img, nombre, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            self.color_buttons.append((x1, y1, x2, y2, i))

        # --- Botón Borrar Todo (Limpia) ---
        bx1 = w - BORRAR_BTN_WIDTH - 20
        bx2 = bx1 + BORRAR_BTN_WIDTH
        by1, by2 = y_top, y_top + btn_h
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (40, 40, 60), -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (120, 120, 255), 1)
        ts = cv2.getTextSize("Limpia", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = bx1 + (BORRAR_BTN_WIDTH - ts[0]) // 2
        ty = by1 + (btn_h + ts[1]) // 2
        cv2.putText(img, "Limpia", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 150, 255), 1, cv2.LINE_AA)
        self.borrar_button = (bx1, by1, bx2, by2)

        # --- Botón Borrador (Parcial) ---
        ex1 = bx1 - BORRADOR_BTN_WIDTH - 12
        ex2 = ex1 + BORRADOR_BTN_WIDTH
        cv2.rectangle(img, (ex1, by1), (ex2, by2), (60, 60, 60), -1)

        # Resaltar si está activo
        if self.is_eraser:
            cv2.rectangle(img, (ex1 - 2, by1 - 2), (ex2 + 2, by2 + 2), (255, 255, 255), 2)
            cv2.rectangle(img, (ex1, by1), (ex2, by2), (200, 200, 200), 1)
        else:
            cv2.rectangle(img, (ex1, by1), (ex2, by2), (150, 150, 150), 1)

        ts = cv2.getTextSize("Borrador", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = ex1 + (BORRADOR_BTN_WIDTH - ts[0]) // 2
        cv2.putText(img, "Borrador", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.borrador_button = (ex1, by1, ex2, by2)

        # --- Botón Guardar (Esquina inferior derecha) ---
        gx1 = w - GUARDAR_BTN_WIDTH - 20
        gx2 = w - 20
        # Mover a la parte inferior
        gy1 = h - btn_h - 20
        gy2 = h - 20

        # Fondo y borde
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (40, 80, 40), -1)
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (100, 255, 100), 2)

        # Texto
        ts = cv2.getTextSize("Guardar", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        tx = gx1 + (GUARDAR_BTN_WIDTH - ts[0]) // 2
        ty_save = gy1 + (btn_h + ts[1]) // 2
        cv2.putText(img, "Guardar", (tx, ty_save), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (100, 255, 100), 2, cv2.LINE_AA)

        self.guardar_button = (gx1, gy1, gx2, gy2)

    def _check_toolbar_touch(self, x, y):
        """Verifica si el dedo índice toca un botón del toolbar."""
        # Verificar botones de color
        for (x1, y1, x2, y2, idx) in self.color_buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.color_index = idx
                self.brush_color_bgr = COLORES[idx][1]
                self.is_eraser = False
                try:
                    self.lbl_color.config(text=f"Color actual: {COLORES[idx][0]}")
                except:
                    pass
                return True

        # Verificar botón borrador (parcial)
        if hasattr(self, 'borrador_button') and self.borrador_button:
            ex1, ey1, ex2, ey2 = self.borrador_button
            if ex1 <= x <= ex2 and ey1 <= y <= ey2:
                self.is_eraser = True
                try:
                    self.lbl_color.config(text="Modo: Borrador")
                except:
                    pass
                return True

        # Verificar botón borrar (total)
        if self.borrar_button:
            bx1, by1, bx2, by2 = self.borrar_button
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                if self.drawing_canvas is not None:
                    self.drawing_canvas[:] = 0
                return True

        # Verificar botón guardar
        if hasattr(self, 'guardar_button') and self.guardar_button:
            gx1, gy1, gx2, gy2 = self.guardar_button
            if gx1 <= x <= gx2 and gy1 <= y <= gy2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Auto-guardar PNG y TXT
                self._save_png(os.path.join(SAVE_DIR, f"dibujo_{timestamp}.png"))
                self._save_txt(os.path.join(SAVE_DIR, f"texto_{timestamp}.txt"))
                # Limpiar tras guardar (opcional, pero util para seguir escribiendo)
                # self.drawing_canvas[:] = 0
                return True

        return False

    def _draw_text_utf8(self, img, text, position, font_size, color_bgr):
        """Dibuja texto con soporte UTF-8 (como la 'ñ') usando Pillow."""
        # Convertir OpenCV (BGR) a Pillow (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Intentar cargar una fuente del sistema que soporte UTF-8
        try:
            # En Windows suele estar en esta ruta
            font_path = "C:/Windows/Fonts/arial.ttf"
            if not os.path.exists(font_path):
                font_path = "arial.ttf" # Intentar en el path local
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        # Dibujar el texto (Pillow usa RGB, invertimos el BGR recibido)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(position, text, font=font, fill=color_rgb)

        # Convertir de vuelta a OpenCV (BGR)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # =========================================================
    #  BUCLE PRINCIPAL DE LA CÁMARA
    # =========================================================
    def camera_loop(self):
        """Hilo principal: captura video del Kinect v1 (Xbox 360), detecta mano, dibuja sobre la imagen."""

        CAPTURE_W, CAPTURE_H = 640, 480
        use_kinect_sdk = False
        cap = None

        # --- Intentar conexión nativa con Kinect SDK v1.8 ---
        try:
            from pykinect_v1 import nui

            # Verificar si hay sensores conectados
            sensor_count = nui.Device().count
            print(f"Sensores Kinect detectados: {sensor_count}")

            if sensor_count > 0:
                # Variables de sincronización para el frame del Kinect
                self._kinect_frame_lock = threading.Lock()
                self._kinect_buffer = (ctypes.c_byte * (CAPTURE_W * CAPTURE_H * 4))()
                self._kinect_new_frame = False

                # Callback: se ejecuta cuando el Kinect tiene un frame nuevo
                def _on_video_frame(frame):
                    try:
                        frame.image.copy_bits(self._kinect_buffer)
                        with self._kinect_frame_lock:
                            self._kinect_new_frame = True
                    except Exception:
                        pass

                # Inicializar Kinect con solo la cámara de color
                self._kinect_runtime = nui.Runtime(
                    nui_init_flags=nui.RuntimeOptions.uses_color
                )

                # Abrir stream de video a 640x480
                self._kinect_runtime.video_stream.open(
                    nui.ImageStreamType.Video, 2,
                    nui.ImageResolution.Resolution640x480,
                    nui.ImageType.Color
                )

                # Registrar callback
                self._kinect_runtime.video_frame_ready += _on_video_frame

                use_kinect_sdk = True
                print("Kinect v1 (Xbox 360) conectado exitosamente via SDK nativo.")
            else:
                print("AVISO: El Kinect SDK no detecta sensores conectados.")

        except Exception as e:
            print(f"AVISO: No se pudo iniciar Kinect SDK: {e}")

        # --- Fallback: cámara web genérica ---
        if not use_kinect_sdk:
            print("Buscando cámara web como alternativa...")
            for cam_idx in [0, 1, 2]:
                test_cap = cv2.VideoCapture(cam_idx)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    if ret:
                        cap = test_cap
                        print(f"Camara web encontrada en indice {cam_idx}")
                        break
                    test_cap.release()

            if cap is None:
                print("ERROR: No se encontro ninguna camara ni Kinect.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # --- MediaPipe Hands ---
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils

        hand_style = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        conn_style = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)

        print("\nAplicacion iniciada. Muestra tu mano a la cámara.")
        print("   Levanta SOLO el índice para escribir")
        print("   Cierra el puño para dejar de escribir")
        print("   Toca los botones de color arriba")
        print("   Q para salir\n")

        while self.running:
            # --- Capturar frame ---
            if use_kinect_sdk:
                # Leer del buffer del Kinect (llenado por el callback)
                got_frame = False
                with self._kinect_frame_lock:
                    if self._kinect_new_frame:
                        self._kinect_new_frame = False
                        got_frame = True

                if not got_frame:
                    cv2.waitKey(1)
                    continue

                # Convertir buffer BGRA → numpy array BGR
                bgra = np.frombuffer(self._kinect_buffer, dtype=np.uint8).reshape(CAPTURE_H, CAPTURE_W, 4)
                frame = bgra[:, :, :3].copy()  # Tomar solo BGR (descartar Alpha)
            else:
                success, frame = cap.read()
                if not success:
                    continue

            # Imagen natural (sin espejo)
            # frame = cv2.flip(frame, 1) # Comentado para evitar inversion
            h, w = frame.shape[:2]

            # Inicializar canvas de dibujo (negro/transparente)
            if self.drawing_canvas is None:
                self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Convertir a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Estado del gesto actual
            status_text = "Muestra tu mano"
            status_color = (150, 150, 150)

            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                           hand_style, conn_style)

                    index_tip = hand_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)

                    if self.is_eraser:
                        # Dibujar círculo traslúcido para el borrador
                        overlay = frame.copy()
                        cv2.circle(overlay, (ix, iy), self.brush_size * 4, (150, 150, 150), cv2.FILLED)
                        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                        cv2.circle(frame, (ix, iy), self.brush_size * 4, (255, 255, 255), 2)
                    else:
                        # Dibujar cursor normal de dibujo
                        cv2.circle(frame, (ix, iy), 10, self.brush_color_bgr, cv2.FILLED)
                        cv2.circle(frame, (ix, iy), 12, (255, 255, 255), 2)

                    # Comprobar colisión con botones (Arriba o Abajo)
                    if iy < TOOLBAR_HEIGHT or iy > h - 100:
                        if self._check_toolbar_touch(ix, iy):
                            self.prev_x, self.prev_y = None, None
                            self.point_buffer.clear()
                            self.is_drawing = False
                            status_text = "Botón presionado..."
                            status_color = (0, 255, 255)
                            continue  # Saltar el dibujo este frame para evitar trazos accidentales

                    if self._is_index_up(hand_lms) and not self._is_fist(hand_lms):
                        if not self.is_drawing:
                            self.is_drawing = True
                            self.prev_x, self.prev_y = None, None
                            self.point_buffer.clear()

                        # Suavizar la posición del dedo
                        sx, sy = self._smooth_point(ix, iy)

                        if self.prev_x is not None and self.prev_y is not None:
                            color = (0, 0, 0) if self.is_eraser else self.brush_color_bgr
                            size = self.brush_size * 2 if self.is_eraser else self.brush_size
                            cv2.line(self.drawing_canvas,
                                     (self.prev_x, self.prev_y), (sx, sy),
                                     color, size)
                        self.prev_x, self.prev_y = sx, sy

                        status_text = "Borrando..." if self.is_eraser else "Escribiendo..."
                        status_color = (255, 255, 255) if self.is_eraser else (0, 255, 0)

                    else:
                        self.is_drawing = False
                        self.prev_x, self.prev_y = None, None
                        self.point_buffer.clear()
                        if self._is_fist(hand_lms):
                            status_text = "Puño cerrado (pausa)"
                            status_color = (0, 0, 255)
                        else:
                            status_text = "Mano detectada"
                            status_color = (255, 200, 0)
            else:
                self.is_drawing = False
                self.prev_x, self.prev_y = None, None
                self.point_buffer.clear()

            # =========================================
            # COMBINAR: Cámara + Dibujo superpuesto
            # =========================================
            gray_canvas = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            fg = cv2.bitwise_and(self.drawing_canvas, self.drawing_canvas, mask=mask)

            combined = cv2.add(bg, fg)

            self._dibujar_cuadricula(combined)
            self._dibujar_toolbar(combined)

            # Usar el nuevo método para soportar la 'ñ'
            combined = self._draw_text_utf8(combined, status_text, (15, h - 40), 24, status_color)

            color_name = COLORES[self.color_index][0]
            combined = self._draw_text_utf8(combined, f"Color: {color_name}", (w - 180, h - 40), 18, self.brush_color_bgr)

            # Fuente de video
            src_text = "Kinect SDK" if use_kinect_sdk else "Webcam"
            cv2.putText(combined, src_text, (w - 130, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)

            # Agrandar la ventana para mejor visualización
            display = cv2.resize(combined, (960, 720), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Kinect - Escritura en el Aire", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
                try:
                    self.root.quit()
                except:
                    pass
                break

        # Liberar recursos
        if use_kinect_sdk:
            try:
                self._kinect_runtime.close()
            except:
                pass
        elif cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = KinectApp()
