import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox
import threading
import torch
import torch.nn as nn
import torchvision.models as models
import easyocr
import os

class GestureRecognitionModel(nn.Module):
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

class KinectApp:
    def __init__(self):
        self.mode = "mouse"
        
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.brush_color_bgr = (0, 0, 0)
        self.brush_size = 5
        self.running = True
        self.prev_x, self.prev_y = 0, 0
        
        try:
            print("Cargando la arquitectura PyTorch basado en EasyOCR...")
            use_gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(['es', 'en'], gpu=use_gpu)
            print("Modelos PyTorch cargados exitosamente.")
        except Exception as e:
            print("Error al cargar el modelo de OCR:", e)
            self.reader = None

        print("\nInstanciando la arquitectura de gestos en PyTorch...")
        self.modelo_gestos = GestureRecognitionModel(num_classes=2)
        print("Arquitectura cargada.")

        self.cam_thread = threading.Thread(target=self.camera_loop)
        self.cam_thread.start()

        self.root = tk.Tk()
        self.root.title("Panel Control Kinect")
        self.root.geometry("300x400")
        
        tk.Label(self.root, text="Opciones de la Aplicación", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        self.btn_mode = tk.Button(self.root, text="Modo Actual: MOUSE", width=25, command=self.toggle_mode, bg="lightblue")
        self.btn_mode.pack(pady=5)
        
        self.btn_color = tk.Button(self.root, text="Cambiar Color de Lápiz", width=25, command=self.change_color)
        self.btn_color.pack(pady=5)
        
        tk.Label(self.root, text="Ancho del Lápiz").pack(pady=(10,0))
        self.scale_size = tk.Scale(self.root, from_=1, to=20, orient=tk.HORIZONTAL, command=self.update_size)
        self.scale_size.set(self.brush_size)
        self.scale_size.pack()
        
        self.btn_clear = tk.Button(self.root, text="Limpiar Pizarra", width=25, command=self.clear_canvas)
        self.btn_clear.pack(pady=5)
        
        self.btn_png = tk.Button(self.root, text="Guardar Pizarra en PNG", width=25, command=self.save_png, bg="lightgreen")
        self.btn_png.pack(pady=5)
        
        self.btn_txt = tk.Button(self.root, text="Pasar a TXT (OCR PyTorch)", width=25, command=self.save_txt, bg="lightgreen")
        self.btn_txt.pack(pady=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def update_size(self, val):
        self.brush_size = int(val)

    def toggle_mode(self):
        if self.mode == "mouse":
            self.mode = "pizarra"
            self.btn_mode.config(text="Modo Actual: PIZARRA", bg="orange")
        else:
            self.mode = "mouse"
            self.btn_mode.config(text="Modo Actual: MOUSE", bg="lightblue")
            
    def change_color(self):
        color = colorchooser.askcolor(title="Seleccionar Color de Lápiz")
        if color[0]:
            r, g, b = [int(x) for x in color[0]]
            self.brush_color_bgr = (b, g, r)

    def clear_canvas(self):
        self.canvas.fill(255)
        
    def save_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Archivos PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.canvas)
            messagebox.showinfo("Éxito", "Imagen guardada exitosamente.")
            
    def save_txt(self):
        if not self.reader:
            messagebox.showerror("Error", "El modelo de OCR (PyTorch) no está disponible.")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Archivos de Texto", "*.txt")])
        if path:
            print("Extrayendo texto usando la red...")
            resultados = self.reader.readtext(self.canvas)
            texto_extraido = "\n".join([res[1] for res in resultados])
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(texto_extraido)
                
            messagebox.showinfo("Texto Extraído", f"El texto capturado es:\n\n{texto_extraido}")

    def on_close(self):
        self.running = False
        self.root.destroy()

    def camera_loop(self):
        # Intentar importar freenect para soporte nativo de Kinect v1
        try:
            import freenect
            self.use_freenect = True
            print("Freenect detectado. Usando Kinect v1 de Xbox 360 de forma nativa.")
        except ImportError:
            self.use_freenect = False
            print("Freenect no encontrado. Buscando el Kinect como cámara web genérica (VideoCapture 0).")
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        mp_draw = mp.solutions.drawing_utils
        
        screen_w, screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False

        while self.running:
            if self.use_freenect:
                try:
                    # Freenect devuelve la imagen en formato RGB (numpy array dimensionalidad (480, 640, 3))
                    video_tensor, _ = freenect.sync_get_video()
                    if video_tensor is None:
                        continue
                    img_rgb = video_tensor.copy()
                    # Pasamos a BGR para que OpenCV pueda procesar la visualización
                    img = cv2.cvtColor(video_tensor, cv2.COLOR_RGB2BGR)
                    success = True
                except Exception as e:
                    print(f"Error de Freenect al leer imagen del Kinect: {e}")
                    success = False
            else:
                success, img = cap.read()
                if success:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if not success:
                continue
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.flip(img_rgb, 1)
            results = hands.process(img_rgb)
            
            lmlist = []
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                    for id, lm in enumerate(hand_lms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmlist.append((id, cx, cy))
            
            if len(lmlist) != 0:
                x1, y1 = lmlist[8][1], lmlist[8][2]
                x2, y2 = lmlist[4][1], lmlist[4][2]
                dist = math.hypot(x2 - x1, y2 - y1)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                if self.mode == "mouse":
                    screen_x = np.interp(x1, (50, 590), (0, screen_w))
                    screen_y = np.interp(y1, (50, 430), (0, screen_h))
                    
                    try:
                        pyautogui.moveTo(screen_x, screen_y)
                    except Exception:
                        pass
                    
                    if dist < 40:
                        pyautogui.click()
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                
                elif self.mode == "pizarra":
                    cv2.circle(img, (x1, y1), 10, self.brush_color_bgr, cv2.FILLED)
                    if dist < 40:
                        if self.prev_x == 0 and self.prev_y == 0:
                            self.prev_x, self.prev_y = x1, y1
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), self.brush_color_bgr, self.brush_size)
                        self.prev_x, self.prev_y = x1, y1
                    else:
                        self.prev_x, self.prev_y = 0, 0
            else:
                self.prev_x, self.prev_y = 0, 0
                
            cv2.imshow("Kinect - Deteccion de Mano", img)
            if self.mode == "pizarra":
                cv2.imshow("Pizarra Virtual OCR", self.canvas)
            else:
                try: cv2.destroyWindow("Pizarra Virtual OCR")
                except: pass
            
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
                self.root.quit()
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = KinectApp()
