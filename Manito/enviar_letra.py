import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import websocket
import threading
import json
import os

ESP32_IP = "192.168.1.50"
CALIBRACION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# ── Colors ────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
BG_CARD   = "#16213e"
ACCENT    = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT      = "#eaeaea"
TEXT_DIM  = "#8a8a9a"

VOWEL_COLORS = {
    "A": "#e94560", "E": "#f5a623", "I": "#7ed321",
    "O": "#4a90d9", "U": "#9b59b6",
}
CONSONANT_COLOR = "#2c3e50"

# ── Finger order (natural: 4 fingers first, thumb last) ──────────────
DEDOS     = ["idx", "med", "anu", "men", "pd",   "pp"]
DEDOS_NOM = ["Índice", "Medio", "Anular", "Meñique", "Pulgar Dedo", "Pulgar Palma"]

# Full LSC alphabet (A-Z + Ñ)
LETRAS = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")

LETRAS_CON_MOVIMIENTO = {"G", "J", "Ñ", "S", "Z"}

def _color_letra(letra):
    return VOWEL_COLORS.get(letra, CONSONANT_COLOR)

DEFAULT_LIMITS = {
    "idx": {"min": 0, "max": 190},
    "med": {"min": 0, "max": 180},
    "anu": {"min": 0, "max": 180},
    "men": {"min": 0, "max": 180},
    "pd":  {"min": 0, "max": 120},
    "pp":  {"min": 0, "max":  90},
}

# Order: [idx, med, anu, men, pd, pp]
DEFAULT_LETRAS = {
    "a": [180, 180, 180, 180,  60,   0],
    "b": [  0,   0,   0,   0, 120,  50],
    "c": [ 90,  90,  90,  90,  45,   0],
    "d": [  0, 180, 180, 180,  60,  25],
    "e": [ 90,  90,  95,  90,   0,  37],
    "f": [120,   0,   0,   0,  90,  30],
    "g": [  0, 180, 180, 180,   0,   0],
    "h": [  0,   0, 180, 180, 120,  50],
    "i": [180, 180, 180,   0,  60,  25],
    "j": [180, 180, 180,   0,  60,  25],
    "k": [  0,   0, 180, 180,  45,  25],
    "l": [  0, 180, 180, 180,   0,   0],
    "m": [ 90,  90,  90, 180, 120,  50],
    "n": [ 90,  90, 180, 180, 120,  40],
    "ñ": [ 90,  90, 180, 180, 120,  40],
    "o": [120, 120, 120, 120,  60,  25],
    "p": [  0,   0, 180, 180,  45,   0],
    "q": [180, 180, 180, 180,   0,   0],
    "r": [  0,   0, 180, 180, 120,  50],
    "s": [ 30,  30,  30,  30,   0,   0],
    "t": [180, 180, 180, 180,  60,  30],
    "u": [  0,   0, 180, 180,  60,  25],
    "v": [  0,   0, 180, 180, 120,  50],
    "w": [  0,   0,   0, 180, 120,  50],
    "x": [ 90, 180, 180, 180,  60,  25],
    "y": [180, 180, 180,   0,   0,   0],
    "z": [  0, 180, 180, 180, 120,  50],
}

DEFAULT_MOVEMENTS = {
    "g": [{"finger": "idx", "to": 180, "repeat": 3, "delay_ms": 200}],
    "j": [{"finger": "men", "to": 40,  "repeat": 1, "delay_ms": 150}],
    "ñ": [{"finger": "idx", "to": -90, "repeat": 2, "delay_ms": 150},
          {"finger": "med", "to": -90, "repeat": 2, "delay_ms": 150}],
    "s": [{"finger": "pp",  "to": 50,  "repeat": 1, "delay_ms": 150},
          {"finger": "idx", "to": 30,  "repeat": 1, "delay_ms": 100},
          {"finger": "pp",  "to": 50,  "repeat": 1, "delay_ms": 150},
          {"finger": "idx", "to": 30,  "repeat": 1, "delay_ms": 100},
          {"finger": "pp",  "to": 50,  "repeat": 1, "delay_ms": 150}],
    "z": [{"finger": "anu", "to": -30, "repeat": 2, "delay_ms": 150},
          {"finger": "med", "to": -30, "repeat": 2, "delay_ms": 150}],
}

# Fingers that close before thumb (per letter)
DEFAULT_PRE_THUMB = {
    "m": ["men"],
    "n": ["anu", "men"],
    "ñ": ["anu", "men"],
}

FINGER_INDEX_MAP = {k: i for i, k in enumerate(DEDOS)}


# ── Default settings ──────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "delay_between_fingers_ms": 50,
    "delay_between_letters_ms": 1500,
    "delay_before_movements_ms": 200,
}


class ManitoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Manito - Lengua de Señas Colombiana")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.ws          = None
        self.conectado   = False
        self._escuchando = False

        # Letter queue
        self._cola = []        # list of letters pending
        self._cola_activa = False
        self._cola_cancelada = False

        self.settings = dict(DEFAULT_SETTINGS)
        self.limits  = {k: dict(v) for k, v in DEFAULT_LIMITS.items()}
        self.letras  = {k: list(v) for k, v in DEFAULT_LETRAS.items()}
        self.movimientos = {k: [dict(m) for m in v] for k, v in DEFAULT_MOVEMENTS.items()}
        self.pre_thumb = {k: list(v) for k, v in DEFAULT_PRE_THUMB.items()}

        self._cargar_config_local()
        self._crear_ui()
        self.root.after(100, self._reconectar)

    # ── Local persistence (config.json) ────────────────────────────────

    def _cargar_config_local(self):
        if not os.path.exists(CALIBRACION_FILE):
            return
        try:
            with open(CALIBRACION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "settings" in data:
                for k, v in data["settings"].items():
                    if k in self.settings and isinstance(v, (int, float)):
                        self.settings[k] = int(v)
            if "limits" in data:
                for k, v in data["limits"].items():
                    if k in self.limits:
                        self.limits[k].update(v)
            if "letters" in data:
                # New format: {"letters": {"a": {"pos": [...], "movements": [...]}}}
                for k, v in data["letters"].items():
                    if k in self.letras and isinstance(v, dict):
                        if "pos" in v and isinstance(v["pos"], list) and len(v["pos"]) == 6:
                            self.letras[k] = list(v["pos"])
                        if "movements" in v and isinstance(v["movements"], list):
                            self.movimientos[k] = [dict(m) for m in v["movements"]]
                        if "pre_thumb" in v and isinstance(v["pre_thumb"], list):
                            self.pre_thumb[k] = list(v["pre_thumb"])
            elif "letras" in data:
                # Legacy format: {"letras": {"a": [...]}} or {"letras": {"a": {"pos":..}}}
                for k, v in data["letras"].items():
                    if k in self.letras:
                        if isinstance(v, list) and len(v) == 6:
                            self.letras[k] = list(v)
                        elif isinstance(v, dict):
                            if "pos" in v and isinstance(v["pos"], list) and len(v["pos"]) == 6:
                                self.letras[k] = list(v["pos"])
                            if "movements" in v and isinstance(v["movements"], list):
                                self.movimientos[k] = [dict(m) for m in v["movements"]]
        except Exception:
            pass

    def _guardar_config_local(self):
        try:
            letters_data = {}
            for k in self.letras:
                letters_data[k] = {
                    "pos": self.letras[k],
                    "pre_thumb": self.pre_thumb.get(k, []),
                    "movements": self.movimientos.get(k, []),
                }
            with open(CALIBRACION_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "settings": self.settings,
                    "limits": self.limits,
                    "letters": letters_data
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log_seguro(f"Error guardando config local: {e}")

    def _aplicar_config(self, data):
        """Apply config received from ESP32 and update UI."""
        if "settings" in data:
            for k, v in data["settings"].items():
                if k in self.settings:
                    self.settings[k] = int(v)
            if hasattr(self, "_settings_vars"):
                self._actualizar_settings_ui()
        if "limits" in data:
            for k, v in data["limits"].items():
                if k in self.limits:
                    self.limits[k]["min"] = v.get("min", self.limits[k]["min"])
                    self.limits[k]["max"] = v.get("max", self.limits[k]["max"])
        if "letras" in data:
            for k, v in data["letras"].items():
                if k in self.letras:
                    if isinstance(v, dict):
                        if "pos" in v and isinstance(v["pos"], list) and len(v["pos"]) == 6:
                            self.letras[k] = list(v["pos"])
                        if "movements" in v and isinstance(v["movements"], list):
                            self.movimientos[k] = [dict(m) for m in v["movements"]]
                        if "pre_thumb" in v and isinstance(v["pre_thumb"], list):
                            self.pre_thumb[k] = list(v["pre_thumb"])
                    elif isinstance(v, list) and len(v) == 6:
                        self.letras[k] = list(v)

        if hasattr(self, "_sliders_letra"):
            self._actualizar_sliders_letra()
        if hasattr(self, "_entries_limites"):
            self._actualizar_entries_limites()

        self._guardar_config_local()
        self._log_seguro("Config recibida y guardada localmente.")

    # ── Main UI ───────────────────────────────────────────────────────

    def _crear_ui(self):
        # Header
        frame_h = tk.Frame(self.root, bg=BG)
        frame_h.pack(pady=(16, 4))
        tk.Label(frame_h, text="*", font=("Segoe UI", 32, "bold"), fg=HIGHLIGHT, bg=BG).pack()
        tk.Label(frame_h, text="Manito", font=("Segoe UI", 22, "bold"), fg=TEXT, bg=BG).pack()
        tk.Label(frame_h, text="Lengua de Señas Colombiana",
                 font=("Segoe UI", 10), fg=TEXT_DIM, bg=BG).pack()

        # Connection status
        self.lbl_estado = tk.Label(self.root, text="● Desconectado",
                                   font=("Segoe UI", 10), fg="#e74c3c", bg=BG)
        self.lbl_estado.pack(pady=(4, 8))

        # IP entry
        frame_ip = tk.Frame(self.root, bg=BG)
        frame_ip.pack(pady=(0, 8))
        tk.Label(frame_ip, text="IP:", font=("Segoe UI", 10), fg=TEXT_DIM, bg=BG
                 ).pack(side=tk.LEFT, padx=(0, 5))
        self.entry_ip = tk.Entry(frame_ip, font=("Consolas", 11), width=16,
                                 bg=ACCENT, fg=TEXT, insertbackground=TEXT,
                                 relief=tk.FLAT, justify=tk.CENTER)
        self.entry_ip.insert(0, ESP32_IP)
        self.entry_ip.pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(frame_ip, text="Conectar", font=("Segoe UI", 9, "bold"),
                  bg=ACCENT, fg=TEXT, activebackground=HIGHLIGHT, activeforeground=TEXT,
                  relief=tk.FLAT, cursor="hand2",
                  command=self._reconectar).pack(side=tk.LEFT)

        # ── Notebook (tabs) ──
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",     background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=ACCENT, foreground=TEXT,
                        padding=[14, 5], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", HIGHLIGHT)],
                  foreground=[("selected", "white")])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        tab_main   = tk.Frame(self.notebook, bg=BG)
        tab_finger = tk.Frame(self.notebook, bg=BG)
        tab_calib  = tk.Frame(self.notebook, bg=BG)
        tab_lim    = tk.Frame(self.notebook, bg=BG)
        tab_settings = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(tab_main,     text="  Principal  ")
        self.notebook.add(tab_finger,   text="  Probar Dedo  ")
        self.notebook.add(tab_calib,    text="  Calibración  ")
        self.notebook.add(tab_lim,      text="  Límites  ")
        self.notebook.add(tab_settings, text="  Ajustes  ")

        self._crear_tab_principal(tab_main)
        self._crear_tab_probar_dedo(tab_finger)
        self._crear_tab_calibracion(tab_calib)
        self._crear_tab_limites(tab_lim)
        self._crear_tab_settings(tab_settings)

    # ── Main Tab ──────────────────────────────────────────────────────

    def _crear_tab_principal(self, parent):
        # Scrollable frame
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # ── Text input for queuing letters ──
        tk.Label(scroll_frame, text="ESCRIBIR TEXTO", font=("Segoe UI", 10, "bold"),
                 fg=TEXT_DIM, bg=BG).pack(pady=(12, 5))

        frame_text = tk.Frame(scroll_frame, bg=BG)
        frame_text.pack(pady=(0, 5), padx=15, fill=tk.X)

        self.entry_texto = tk.Entry(
            frame_text, font=("Consolas", 14), bg=ACCENT, fg=TEXT,
            insertbackground=TEXT, relief=tk.FLAT, justify=tk.LEFT)
        self.entry_texto.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=4)
        self.entry_texto.bind("<Return>", lambda e: self._enviar_texto())

        tk.Button(frame_text, text="Enviar", font=("Segoe UI", 11, "bold"),
                  bg="#3498db", fg="white", activebackground="#2980b9",
                  activeforeground="white", relief=tk.FLAT, cursor="hand2",
                  command=self._enviar_texto).pack(side=tk.LEFT, padx=(0, 4))

        self.btn_cancelar = tk.Button(
            frame_text, text="CANCELAR", font=("Segoe UI", 11, "bold"),
            bg="#e74c3c", fg="white", activebackground="#c0392b",
            activeforeground="white", relief=tk.FLAT, cursor="hand2",
            state=tk.DISABLED, command=self._cancelar_cola)
        self.btn_cancelar.pack(side=tk.LEFT)

        # Queue status label
        self.lbl_cola = tk.Label(scroll_frame, text="",
                                  font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG)
        self.lbl_cola.pack(pady=(0, 5))

        # Separator
        tk.Frame(scroll_frame, bg=ACCENT, height=1).pack(fill=tk.X, padx=30, pady=6)

        # ── All letters in one grid (A-Z + Ñ) ──
        tk.Label(scroll_frame, text="ABECEDARIO LSC", font=("Segoe UI", 10, "bold"),
                 fg=TEXT_DIM, bg=BG).pack(pady=(5, 5))

        frame_letras = tk.Frame(scroll_frame, bg=BG)
        frame_letras.pack(pady=5, padx=10)

        COLS = 7
        for i, letra in enumerate(LETRAS):
            row_idx = i // COLS
            col_idx = i % COLS
            label_text = f"~{letra}" if letra in LETRAS_CON_MOVIMIENTO else letra
            color = _color_letra(letra)
            if letra in LETRAS_CON_MOVIMIENTO:
                color = "#1abc9c"
            btn = tk.Button(
                frame_letras, text=label_text, font=("Segoe UI", 14, "bold"),
                width=4, height=1, bg=color, fg="white",
                activebackground=self._oscurecer(color), activeforeground="white",
                relief=tk.FLAT, cursor="hand2",
                command=lambda lt=letra: self._encolar_letra(lt.lower()))
            btn.grid(row=row_idx, column=col_idx, padx=4, pady=4)
            btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=self._aclarar(c)))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))

        # Separator
        tk.Frame(scroll_frame, bg=ACCENT, height=1).pack(fill=tk.X, padx=30, pady=8)

        # Controls
        tk.Label(scroll_frame, text="CONTROLES", font=("Segoe UI", 10, "bold"),
                 fg=TEXT_DIM, bg=BG).pack(pady=(0, 5))
        frame_ctrl = tk.Frame(scroll_frame, bg=BG)
        frame_ctrl.pack(pady=(0, 8))
        for txt, color, active, cmd in [
            ("[+] Abrir mano",  "#2ecc71", "#27ae60", "reset"),
            ("[x] Cerrar mano", "#e74c3c", "#c0392b", "cerrar"),
        ]:
            tk.Button(frame_ctrl, text=txt, font=("Segoe UI", 13), width=14,
                      bg=color, fg="white", activebackground=active,
                      activeforeground="white", relief=tk.FLAT, cursor="hand2",
                      command=lambda c=cmd: self._enviar(c)).pack(side=tk.LEFT, padx=8)

        # Separator
        tk.Frame(scroll_frame, bg=ACCENT, height=1).pack(fill=tk.X, padx=30, pady=8)

        # Serial monitor
        tk.Label(scroll_frame, text="SERIAL MONITOR (WiFi)", font=("Segoe UI", 10, "bold"),
                 fg=TEXT_DIM, bg=BG).pack(pady=(0, 4))
        self.txt_log = scrolledtext.ScrolledText(
            scroll_frame, font=("Consolas", 9), width=52, height=8,
            bg="#0d1117", fg="#58a6ff", insertbackground=TEXT,
            relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD)
        self.txt_log.pack(padx=15, pady=(0, 4))
        tk.Button(scroll_frame, text="Limpiar log", font=("Segoe UI", 9),
                  bg=ACCENT, fg=TEXT_DIM, activebackground=HIGHLIGHT,
                  activeforeground=TEXT, relief=tk.FLAT, cursor="hand2",
                  command=self._limpiar_log).pack(pady=(0, 12))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel scroll
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    # ── Letter queue ──────────────────────────────────────────────────

    def _encolar_letra(self, letra):
        self._cola.append(letra)
        self._actualizar_lbl_cola()
        if not self._cola_activa:
            self._procesar_cola()

    def _enviar_texto(self):
        texto = self.entry_texto.get().strip().upper()
        self.entry_texto.delete(0, tk.END)
        if not texto:
            return
        letras_validas = set(l.lower() for l in LETRAS)
        for ch in texto:
            ch_lower = ch.lower()
            if ch_lower in letras_validas:
                self._cola.append(ch_lower)
        self._actualizar_lbl_cola()
        if not self._cola_activa:
            self._procesar_cola()

    def _cancelar_cola(self):
        self._cola.clear()
        self._cola_cancelada = True
        self._actualizar_lbl_cola()
        self._agregar_log(">>> Cola cancelada")

    def _procesar_cola(self):
        if not self._cola or self._cola_cancelada:
            self._cola_activa = False
            self._cola_cancelada = False
            self.btn_cancelar.config(state=tk.DISABLED)
            self._actualizar_lbl_cola()
            return

        self._cola_activa = True
        self.btn_cancelar.config(state=tk.NORMAL)
        letra = self._cola.pop(0)
        self._actualizar_lbl_cola()
        self._enviar(letra)

        # Schedule next letter (configurable delay for servo movement)
        self.root.after(self.settings["delay_between_letters_ms"], self._procesar_cola)

    def _actualizar_lbl_cola(self):
        if self._cola:
            pending = " ".join(l.upper() for l in self._cola[:20])
            if len(self._cola) > 20:
                pending += " ..."
            self.lbl_cola.config(text=f"Cola: {pending}", fg="#f5a623")
        else:
            self.lbl_cola.config(text="", fg=TEXT_DIM)

    # ── Per-Finger Test Tab ────────────────────────────────────────────

    def _crear_tab_probar_dedo(self, parent):
        tk.Label(parent, text="Probar Dedo Individual",
                 font=("Segoe UI", 12, "bold"), fg=TEXT, bg=BG).pack(pady=(16, 4))
        tk.Label(parent, text="Mueve un slider para controlar cada dedo en tiempo real",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG).pack(pady=(0, 12))

        frame_dedos = tk.Frame(parent, bg=BG)
        frame_dedos.pack(padx=20, fill=tk.X)

        for d, (key, nom) in enumerate(zip(DEDOS, DEDOS_NOM)):
            row = tk.Frame(frame_dedos, bg=BG)
            row.pack(fill=tk.X, pady=4)

            tk.Label(row, text=f"{nom}:", font=("Segoe UI", 10),
                     fg=TEXT, bg=BG, width=14, anchor="w").pack(side=tk.LEFT)

            lbl_val = tk.Label(row, text="0",
                               font=("Consolas", 11, "bold"),
                               fg=HIGHLIGHT, bg=BG, width=4)
            lbl_val.pack(side=tk.RIGHT)

            def _on_finger_move(val, finger=key, lv=lbl_val):
                lv.config(text=val)
                self._enviar(f"movefinger|{finger}|{val}")

            slider = tk.Scale(
                row,
                from_=self.limits[key]["min"],
                to=self.limits[key]["max"],
                orient=tk.HORIZONTAL, length=300,
                bg=BG, fg=TEXT, troughcolor=ACCENT, highlightthickness=0,
                sliderrelief=tk.FLAT, showvalue=False,
                command=_on_finger_move)
            slider.pack(side=tk.LEFT, padx=6)

    # ── Calibration Tab ───────────────────────────────────────────────

    def _crear_tab_calibracion(self, parent):
        tk.Label(parent, text="Ajusta la posición de cada letra",
                 font=("Segoe UI", 11, "bold"), fg=TEXT, bg=BG).pack(pady=(14, 4))
        tk.Label(parent, text="Usa sliders o botones +/− → Probar → Guardar",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG).pack(pady=(0, 8))

        # Letter selector
        frame_sel = tk.Frame(parent, bg=BG)
        frame_sel.pack(pady=(0, 4))
        tk.Label(frame_sel, text="Letra:", font=("Segoe UI", 11),
                 fg=TEXT_DIM, bg=BG).pack(side=tk.LEFT, padx=(0, 8))

        self._letra_var = tk.StringVar(value="a")

        frame_letras_sel = tk.Frame(parent, bg=BG)
        frame_letras_sel.pack(pady=(0, 8))

        COLS_SEL = 14
        for i, letra in enumerate(LETRAS):
            letra_lower = letra.lower()
            color = VOWEL_COLORS.get(letra, "#3498db")
            rb = tk.Radiobutton(
                frame_letras_sel, text=letra, variable=self._letra_var,
                value=letra_lower, font=("Segoe UI", 11, "bold"),
                bg=BG, fg=color, activebackground=BG, selectcolor=ACCENT,
                indicatoron=False, width=2, relief=tk.FLAT,
                command=self._actualizar_sliders_letra)
            rb.grid(row=i // COLS_SEL, column=i % COLS_SEL, padx=2, pady=2)

        # Sliders per finger with +/- buttons
        frame_sliders = tk.Frame(parent, bg=BG)
        frame_sliders.pack(padx=20, fill=tk.X)

        self._sliders_letra = []
        self._slider_vars   = []
        self._slider_labels = []

        letra0 = self.letras["a"]
        for d, (key, nom) in enumerate(zip(DEDOS, DEDOS_NOM)):
            row = tk.Frame(frame_sliders, bg=BG)
            row.pack(fill=tk.X, pady=3)

            tk.Label(row, text=f"{nom}:", font=("Segoe UI", 10),
                     fg=TEXT, bg=BG, width=14, anchor="w").pack(side=tk.LEFT)

            # Minus button
            btn_minus = tk.Button(
                row, text="−", font=("Segoe UI", 11, "bold"), width=2,
                bg="#e74c3c", fg="white", activebackground="#c0392b",
                relief=tk.FLAT, cursor="hand2",
                command=lambda idx=d: self._ajustar_slider(idx, -5))
            btn_minus.pack(side=tk.LEFT, padx=(4, 2))

            var = tk.IntVar(value=letra0[d])
            self._slider_vars.append(var)

            lbl_val = tk.Label(row, text=str(letra0[d]),
                               font=("Consolas", 11, "bold"),
                               fg=HIGHLIGHT, bg=BG, width=4)

            slider = tk.Scale(
                row,
                from_=self.limits[key]["min"],
                to=self.limits[key]["max"],
                orient=tk.HORIZONTAL, variable=var, length=220,
                bg=BG, fg=TEXT, troughcolor=ACCENT, highlightthickness=0,
                sliderrelief=tk.FLAT, showvalue=False,
                command=lambda val, lv=lbl_val: lv.config(text=val))
            slider.pack(side=tk.LEFT, padx=4)
            self._sliders_letra.append(slider)

            lbl_val.pack(side=tk.LEFT, padx=(4, 2))
            self._slider_labels.append(lbl_val)

            # Plus button
            btn_plus = tk.Button(
                row, text="+", font=("Segoe UI", 11, "bold"), width=2,
                bg="#2ecc71", fg="white", activebackground="#27ae60",
                relief=tk.FLAT, cursor="hand2",
                command=lambda idx=d: self._ajustar_slider(idx, 5))
            btn_plus.pack(side=tk.LEFT, padx=(2, 4))

        # Buttons
        frame_btns = tk.Frame(parent, bg=BG)
        frame_btns.pack(pady=12)
        tk.Button(frame_btns, text=">>  Probar", font=("Segoe UI", 11),
                  bg="#3498db", fg="white", activebackground="#2980b9",
                  activeforeground="white", relief=tk.FLAT, cursor="hand2",
                  command=self._probar_letra).pack(side=tk.LEFT, padx=6)
        tk.Button(frame_btns, text="[Save]  Guardar en ESP32", font=("Segoe UI", 11),
                  bg="#27ae60", fg="white", activebackground="#219a52",
                  activeforeground="white", relief=tk.FLAT, cursor="hand2",
                  command=self._guardar_letra_esp32).pack(side=tk.LEFT, padx=6)
        tk.Button(frame_btns, text="↓  Cargar desde ESP32", font=("Segoe UI", 11),
                  bg=ACCENT, fg=TEXT, activebackground=HIGHLIGHT,
                  activeforeground=TEXT, relief=tk.FLAT, cursor="hand2",
                  command=lambda: self._enviar("getconfig")).pack(side=tk.LEFT, padx=6)

        # ── Movements section ──
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill=tk.X, padx=20, pady=8)
        tk.Label(parent, text="MOVIMIENTOS ESPECIALES",
                 font=("Segoe UI", 10, "bold"), fg=TEXT_DIM, bg=BG).pack(pady=(0, 4))
        tk.Label(parent, text="Repeticiones de dedos (ej: G recoge índice N veces)",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG).pack(pady=(0, 6))

        self._mov_frame = tk.Frame(parent, bg=BG)
        self._mov_frame.pack(padx=20, fill=tk.X)
        self._mov_widgets = []  # list of dicts with widgets per movement row

        frame_mov_btns = tk.Frame(parent, bg=BG)
        frame_mov_btns.pack(pady=8)
        tk.Button(frame_mov_btns, text="+ Agregar Movimiento", font=("Segoe UI", 10),
                  bg="#1abc9c", fg="white", activebackground="#16a085",
                  relief=tk.FLAT, cursor="hand2",
                  command=self._agregar_movimiento).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_mov_btns, text="− Quitar Último", font=("Segoe UI", 10),
                  bg="#e74c3c", fg="white", activebackground="#c0392b",
                  relief=tk.FLAT, cursor="hand2",
                  command=self._quitar_ultimo_movimiento).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_mov_btns, text="[Save] Guardar Movimientos", font=("Segoe UI", 10),
                  bg="#27ae60", fg="white", activebackground="#219a52",
                  relief=tk.FLAT, cursor="hand2",
                  command=self._guardar_movimientos_esp32).pack(side=tk.LEFT, padx=4)

        self._actualizar_ui_movimientos()

        # ── Pre-thumb section ──
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill=tk.X, padx=20, pady=8)
        tk.Label(parent, text="ORDEN PRE-PULGAR",
                 font=("Segoe UI", 10, "bold"), fg=TEXT_DIM, bg=BG).pack(pady=(0, 4))
        tk.Label(parent, text="Dedos que cierran ANTES del pulgar (ej: M cierra meñique primero)",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG).pack(pady=(0, 6))

        frame_pt = tk.Frame(parent, bg=BG)
        frame_pt.pack(padx=20)
        self._pt_vars = {}
        for key, nom in zip(DEDOS[:4], DEDOS_NOM[:4]):
            var = tk.BooleanVar(value=False)
            self._pt_vars[key] = var
            tk.Checkbutton(
                frame_pt, text=nom, variable=var,
                font=("Segoe UI", 10), fg=TEXT, bg=BG, selectcolor=ACCENT,
                activebackground=BG, activeforeground=TEXT
            ).pack(side=tk.LEFT, padx=6)

        tk.Button(parent, text="[Save] Guardar Pre-Pulgar", font=("Segoe UI", 10),
                  bg="#27ae60", fg="white", activebackground="#219a52",
                  relief=tk.FLAT, cursor="hand2",
                  command=self._guardar_prethumb_esp32).pack(pady=6)

        self._actualizar_ui_prethumb()

    # ── Limits Tab ────────────────────────────────────────────────────

    def _crear_tab_limites(self, parent):
        tk.Label(parent, text="Límites de movimiento por dedo",
                 font=("Segoe UI", 12, "bold"), fg=TEXT, bg=BG).pack(pady=(16, 4))
        tk.Label(parent, text="Rango físico del servo: 0 – 180 grados",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG).pack(pady=(0, 12))

        frame_grid = tk.Frame(parent, bg=BG)
        frame_grid.pack(padx=24)

        # Header
        for col, txt in enumerate(["Dedo", "Mínimo", "Máximo"]):
            tk.Label(frame_grid, text=txt, font=("Segoe UI", 10, "bold"),
                     fg=TEXT_DIM, bg=BG, width=12).grid(
                row=0, column=col, padx=10, pady=(0, 6))

        self._entries_limites = {}
        for d, (key, nom) in enumerate(zip(DEDOS, DEDOS_NOM)):
            row = d + 1
            tk.Label(frame_grid, text=nom, font=("Segoe UI", 10),
                     fg=TEXT, bg=BG, anchor="w", width=14).grid(
                row=row, column=0, padx=10, pady=5, sticky="w")

            var_min = tk.IntVar(value=self.limits[key]["min"])
            var_max = tk.IntVar(value=self.limits[key]["max"])
            self._entries_limites[key] = (var_min, var_max)

            for col, var in enumerate([var_min, var_max], start=1):
                tk.Spinbox(
                    frame_grid, textvariable=var, from_=0, to=180,
                    increment=1, width=7, font=("Consolas", 11),
                    bg=ACCENT, fg=TEXT, insertbackground=TEXT,
                    buttonbackground=ACCENT, relief=tk.FLAT, justify=tk.CENTER,
                ).grid(row=row, column=col, padx=10, pady=5)

        # Buttons
        frame_btns = tk.Frame(parent, bg=BG)
        frame_btns.pack(pady=16)
        tk.Button(frame_btns, text="[Save]  Guardar limites en ESP32",
                  font=("Segoe UI", 11), bg="#27ae60", fg="white",
                  activebackground="#219a52", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2",
                  command=self._guardar_limites_esp32).pack(side=tk.LEFT, padx=6)
        tk.Button(frame_btns, text="↓  Cargar desde ESP32",
                  font=("Segoe UI", 11), bg=ACCENT, fg=TEXT,
                  activebackground=HIGHLIGHT, activeforeground=TEXT,
                  relief=tk.FLAT, cursor="hand2",
                  command=lambda: self._enviar("getconfig")).pack(side=tk.LEFT, padx=6)

    # ── Widget updates ────────────────────────────────────────────────

    def _actualizar_sliders_letra(self):
        letra = self._letra_var.get()
        vals  = self.letras.get(letra, [0] * 6)
        for d, (slider, var, lbl) in enumerate(
                zip(self._sliders_letra, self._slider_vars, self._slider_labels)):
            key = DEDOS[d]
            slider.config(from_=self.limits[key]["min"], to=self.limits[key]["max"])
            var.set(vals[d])
            lbl.config(text=str(vals[d]))
        if hasattr(self, "_mov_frame"):
            self._actualizar_ui_movimientos()
        if hasattr(self, "_pt_vars"):
            self._actualizar_ui_prethumb()

    def _actualizar_entries_limites(self):
        for key, (var_min, var_max) in self._entries_limites.items():
            var_min.set(self.limits[key]["min"])
            var_max.set(self.limits[key]["max"])

    def _ajustar_slider(self, idx, delta):
        var = self._slider_vars[idx]
        new_val = max(0, min(180, var.get() + delta))
        var.set(new_val)
        self._slider_labels[idx].config(text=str(new_val))

    # ── Movement UI helpers ───────────────────────────────────────────

    def _actualizar_ui_movimientos(self):
        for w in self._mov_widgets:
            w["frame"].destroy()
        self._mov_widgets.clear()

        letra = self._letra_var.get()
        movs = self.movimientos.get(letra, [])
        for mov in movs:
            self._crear_fila_movimiento(mov)

    def _crear_fila_movimiento(self, mov_data=None):
        if mov_data is None:
            mov_data = {"finger": "idx", "to": 90, "repeat": 1, "delay_ms": 150}

        row = tk.Frame(self._mov_frame, bg=BG_CARD, relief=tk.FLAT)
        row.pack(fill=tk.X, pady=2, padx=4)

        # Finger selector
        tk.Label(row, text="Dedo:", font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_CARD
                 ).pack(side=tk.LEFT, padx=(4, 2))
        finger_var = tk.StringVar(value=mov_data.get("finger", "idx"))
        finger_menu = tk.OptionMenu(row, finger_var, *DEDOS)
        finger_menu.config(bg=ACCENT, fg=TEXT, font=("Segoe UI", 9),
                           highlightthickness=0, relief=tk.FLAT)
        finger_menu.pack(side=tk.LEFT, padx=2)

        # Delta angle
        tk.Label(row, text="Δ°:", font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_CARD
                 ).pack(side=tk.LEFT, padx=(6, 2))
        to_var = tk.IntVar(value=mov_data.get("to", 90))
        tk.Spinbox(row, textvariable=to_var, from_=-180, to=180, width=5,
                   font=("Consolas", 10), bg=ACCENT, fg=TEXT, relief=tk.FLAT,
                   justify=tk.CENTER).pack(side=tk.LEFT, padx=2)

        # Repeat
        tk.Label(row, text="Rep:", font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_CARD
                 ).pack(side=tk.LEFT, padx=(6, 2))
        rep_var = tk.IntVar(value=mov_data.get("repeat", 1))
        tk.Spinbox(row, textvariable=rep_var, from_=1, to=20, width=3,
                   font=("Consolas", 10), bg=ACCENT, fg=TEXT, relief=tk.FLAT,
                   justify=tk.CENTER).pack(side=tk.LEFT, padx=2)

        # Delay
        tk.Label(row, text="ms:", font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_CARD
                 ).pack(side=tk.LEFT, padx=(6, 2))
        delay_var = tk.IntVar(value=mov_data.get("delay_ms", 150))
        tk.Spinbox(row, textvariable=delay_var, from_=50, to=2000, increment=50,
                   width=5, font=("Consolas", 10), bg=ACCENT, fg=TEXT,
                   relief=tk.FLAT, justify=tk.CENTER).pack(side=tk.LEFT, padx=2)

        self._mov_widgets.append({
            "frame": row,
            "finger": finger_var,
            "to": to_var,
            "repeat": rep_var,
            "delay": delay_var,
        })

    def _agregar_movimiento(self):
        if len(self._mov_widgets) >= 8:
            self._agregar_log("Máximo 8 movimientos por letra")
            return
        self._crear_fila_movimiento()

    def _quitar_ultimo_movimiento(self):
        if self._mov_widgets:
            self._mov_widgets[-1]["frame"].destroy()
            self._mov_widgets.pop()

    def _guardar_movimientos_esp32(self):
        letra = self._letra_var.get()
        movs = []
        parts = []
        for w in self._mov_widgets:
            finger_key = w["finger"].get()
            fi = FINGER_INDEX_MAP.get(finger_key, 0)
            to_val = w["to"].get()
            rep = w["repeat"].get()
            dl = w["delay"].get()
            movs.append({"finger": finger_key, "to": to_val, "repeat": rep, "delay_ms": dl})
            parts.append(f"{fi},{to_val},{rep},{dl}")

        self.movimientos[letra] = movs
        self._guardar_config_local()

        # Send to ESP32: setmovements|letra|fi,to,rep,del;fi,to,rep,del
        data_str = ";".join(parts)
        cmd = f"setmovements|{letra}|{data_str}"
        self._enviar(cmd)
        self._agregar_log(f"[Save] Movimientos {letra.upper()} guardados ({len(movs)})")

    # ── Pre-thumb helpers ─────────────────────────────────────────────

    def _actualizar_ui_prethumb(self):
        letra = self._letra_var.get()
        pt = self.pre_thumb.get(letra, [])
        for key, var in self._pt_vars.items():
            var.set(key in pt)

    def _guardar_prethumb_esp32(self):
        letra = self._letra_var.get()
        selected = [k for k, v in self._pt_vars.items() if v.get()]
        self.pre_thumb[letra] = selected
        self._guardar_config_local()
        cmd = f"setprethumb|{letra}|{','.join(selected)}"
        self._enviar(cmd)
        self._agregar_log(f"[Save] Pre-pulgar {letra.upper()}: {selected or 'ninguno'}")

    # ── Calibration actions ───────────────────────────────────────────

    def _probar_letra(self):
        vals = [v.get() for v in self._slider_vars]
        self._enviar("preview|" + ",".join(str(v) for v in vals))

    def _guardar_letra_esp32(self):
        letra = self._letra_var.get()
        vals  = [v.get() for v in self._slider_vars]
        self.letras[letra] = vals
        self._guardar_config_local()
        self._enviar(f"setletra|{letra}|" + ",".join(str(v) for v in vals))

    def _guardar_limites_esp32(self):
        for key, (var_min, var_max) in self._entries_limites.items():
            mn = var_min.get()
            mx = var_max.get()
            if mn > mx:
                nom = DEDOS_NOM[DEDOS.index(key)]
                messagebox.showerror("Error de límites",
                                     f"{nom}: el mínimo ({mn}) es mayor que el máximo ({mx}).")
                return
            self.limits[key]["min"] = mn
            self.limits[key]["max"] = mx

        self._actualizar_sliders_letra()
        self._guardar_config_local()

        for key in DEDOS:
            mn = self.limits[key]["min"]
            mx = self.limits[key]["max"]
            self._enviar(f"setlimits|{key}|{mn}|{mx}")

    # ── Settings Tab ─────────────────────────────────────────────────

    def _crear_tab_settings(self, parent):
        tk.Label(parent, text="Ajustes de Tiempos",
                 font=("Segoe UI", 12, "bold"), fg=TEXT, bg=BG).pack(pady=(16, 4))
        tk.Label(parent, text="Controla los delays entre movimientos (ms)",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG).pack(pady=(0, 12))

        frame_grid = tk.Frame(parent, bg=BG)
        frame_grid.pack(padx=24)

        labels = {
            "delay_between_fingers_ms": "Delay entre dedos (ms)",
            "delay_between_letters_ms": "Delay entre letras (ms)",
            "delay_before_movements_ms": "Delay antes de movimientos (ms)",
        }

        self._settings_vars = {}
        for i, (key, label) in enumerate(labels.items()):
            tk.Label(frame_grid, text=label, font=("Segoe UI", 10),
                     fg=TEXT, bg=BG, anchor="w", width=28).grid(
                row=i, column=0, padx=10, pady=6, sticky="w")

            var = tk.IntVar(value=self.settings[key])
            self._settings_vars[key] = var

            tk.Spinbox(
                frame_grid, textvariable=var, from_=10, to=5000,
                increment=10, width=7, font=("Consolas", 11),
                bg=ACCENT, fg=TEXT, insertbackground=TEXT,
                buttonbackground=ACCENT, relief=tk.FLAT, justify=tk.CENTER,
            ).grid(row=i, column=1, padx=10, pady=6)

        frame_btns = tk.Frame(parent, bg=BG)
        frame_btns.pack(pady=16)
        tk.Button(frame_btns, text="[Save] Guardar en ESP32",
                  font=("Segoe UI", 11), bg="#27ae60", fg="white",
                  activebackground="#219a52", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2",
                  command=self._guardar_settings_esp32).pack(side=tk.LEFT, padx=6)

    def _actualizar_settings_ui(self):
        for key, var in self._settings_vars.items():
            var.set(self.settings[key])

    def _guardar_settings_esp32(self):
        for key, var in self._settings_vars.items():
            self.settings[key] = var.get()
        self._guardar_config_local()
        df = self.settings["delay_between_fingers_ms"]
        dl = self.settings["delay_between_letters_ms"]
        dm = self.settings["delay_before_movements_ms"]
        self._enviar(f"setsettings|{df}|{dl}|{dm}")
        self._agregar_log(f"[Save] Settings: fingers={df}ms letters={dl}ms movs={dm}ms")

    # ── Log ───────────────────────────────────────────────────────────

    def _agregar_log(self, texto):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, texto + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _log_seguro(self, texto):
        self.root.after(0, self._agregar_log, texto)

    def _limpiar_log(self):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.config(state=tk.DISABLED)

    # ── WebSocket connection ──────────────────────────────────────────

    def _build_pushconfig_json(self):
        """Build the full config JSON to push to ESP32."""
        letters_data = {}
        for k in self.letras:
            letters_data[k] = {
                "pos": self.letras[k],
                "pre_thumb": self.pre_thumb.get(k, []),
                "movements": self.movimientos.get(k, []),
            }
        return json.dumps({
            "settings": self.settings,
            "limits": self.limits,
            "letters": letters_data,
        }, ensure_ascii=False)

    def _conectar(self):
        ip  = self.entry_ip.get().strip()
        uri = f"ws://{ip}:81"
        try:
            self.ws = websocket.create_connection(uri, timeout=3)
            self.conectado = True
            self.root.after(0, lambda: self.lbl_estado.config(
                text="● Conectado", fg="#2ecc71"))
            self._log_seguro(f"Conectado a {uri}")
            self._iniciar_escucha()
            # Push local config.json as source of truth
            self.root.after(900, self._push_config_to_esp32)
        except Exception as e:
            self.conectado = False
            self.root.after(0, lambda: self.lbl_estado.config(
                text="● Desconectado", fg="#e74c3c"))
            self._log_seguro(f"Error conectando: {e}")

    def _push_config_to_esp32(self):
        """Push local config.json to ESP32 so it survives reflashes."""
        cfg_json = self._build_pushconfig_json()
        self._enviar(f"pushconfig|{cfg_json}")

    def _reconectar(self):
        self._escuchando = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        self.conectado = False
        self.lbl_estado.config(text="● Conectando...", fg="#f5a623")
        self.root.update()
        threading.Thread(target=self._conectar, daemon=True).start()

    def _iniciar_escucha(self):
        self._escuchando = True

        def escuchar():
            while self._escuchando and self.ws:
                try:
                    self.ws.settimeout(1.0)
                    msg = self.ws.recv()
                    if msg:
                        self._procesar_mensaje(msg)
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:
                    if self._escuchando:
                        self.conectado = False
                        self._escuchando = False
                        self.root.after(0, lambda: self.lbl_estado.config(
                            text="● Desconectado", fg="#e74c3c"))
                        self._log_seguro("Conexión perdida.")
                    break

        threading.Thread(target=escuchar, daemon=True).start()

    def _procesar_mensaje(self, msg):
        if msg.startswith("CONFIG:"):
            try:
                data = json.loads(msg[7:])
                self.root.after(0, lambda d=data: self._aplicar_config(d))
            except Exception as e:
                self._log_seguro(f"CONFIG: error al parsear JSON: {e}")
        else:
            self._log_seguro(f"ESP32> {msg}")

    # ── Send ──────────────────────────────────────────────────────────

    def _enviar(self, comando):
        if not self.conectado or not self.ws:
            messagebox.showwarning(
                "Sin conexión",
                "No estás conectado a la Manito.\nVerifica la IP y presiona Conectar.")
            return
        try:
            self.ws.send(comando)
            # Build human-readable label
            etiquetas = {
                "reset": "Abrir mano", "cerrar": "Cerrar mano",
                "getconfig": "Solicitar config",
            }
            # Skip logging for finger moves
            if comando.startswith("movefinger"):
                return
            # Check single letters
            if len(comando) == 1 and comando.isalpha():
                etiqueta = comando.upper()
            elif comando == "ñ":
                etiqueta = "Ñ"
            else:
                etiqueta = etiquetas.get(comando, comando)

            if len(etiqueta) > 50:
                etiqueta = etiqueta[:50] + "…"
            self._agregar_log(f">>> {etiqueta}")
        except Exception as e:
            self.conectado = False
            self._escuchando = False
            self.lbl_estado.config(text="● Desconectado", fg="#e74c3c")
            messagebox.showerror("Error", f"Se perdió la conexión:\n{e}")

    # ── Color utilities ───────────────────────────────────────────────

    @staticmethod
    def _oscurecer(hex_color, factor=0.7):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    @staticmethod
    def _aclarar(hex_color, factor=1.2):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return f"#{min(int(r*factor),255):02x}{min(int(g*factor),255):02x}{min(int(b*factor),255):02x}"


if __name__ == "__main__":
    root = tk.Tk()
    app  = ManitoApp(root)
    root.mainloop()