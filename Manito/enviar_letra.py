import tkinter as tk
from tkinter import messagebox
import websocket
import threading

ESP32_IP = "192.168.1.50"
URI = f"ws://{ESP32_IP}:81"

# ── Colores y estilos ──
BG = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT = "#eaeaea"
TEXT_DIM = "#8a8a9a"
VOWEL_COLORS = {
    "A": "#e94560",
    "E": "#f5a623",
    "I": "#7ed321",
    "O": "#4a90d9",
    "U": "#9b59b6",
}


class ManitoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Manito - Lengua de Señas")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.ws = None
        self.conectado = False

        self._crear_ui()
        self.root.after(100, self._conectar)

    # ── UI ──

    def _crear_ui(self):
        # Título
        frame_titulo = tk.Frame(self.root, bg=BG)
        frame_titulo.pack(pady=(20, 5))

        tk.Label(
            frame_titulo, text="🤟", font=("Segoe UI Emoji", 36), bg=BG
        ).pack()
        tk.Label(
            frame_titulo,
            text="Manito",
            font=("Segoe UI", 24, "bold"),
            fg=TEXT,
            bg=BG,
        ).pack()
        tk.Label(
            frame_titulo,
            text="Lengua de Señas Colombiana",
            font=("Segoe UI", 11),
            fg=TEXT_DIM,
            bg=BG,
        ).pack()

        # Estado de conexión
        self.lbl_estado = tk.Label(
            self.root,
            text="● Desconectado",
            font=("Segoe UI", 10),
            fg="#e74c3c",
            bg=BG,
        )
        self.lbl_estado.pack(pady=(5, 15))

        # IP editable
        frame_ip = tk.Frame(self.root, bg=BG)
        frame_ip.pack(pady=(0, 10))

        tk.Label(
            frame_ip, text="IP:", font=("Segoe UI", 10), fg=TEXT_DIM, bg=BG
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.entry_ip = tk.Entry(
            frame_ip,
            font=("Consolas", 11),
            width=16,
            bg=ACCENT,
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            justify=tk.CENTER,
        )
        self.entry_ip.insert(0, ESP32_IP)
        self.entry_ip.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_conectar = tk.Button(
            frame_ip,
            text="Conectar",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT,
            fg=TEXT,
            activebackground=HIGHLIGHT,
            activeforeground=TEXT,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._reconectar,
        )
        self.btn_conectar.pack(side=tk.LEFT)

        # Sección de vocales
        tk.Label(
            self.root,
            text="VOCALES",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_DIM,
            bg=BG,
        ).pack(pady=(5, 5))

        frame_vocales = tk.Frame(self.root, bg=BG)
        frame_vocales.pack(pady=5)

        for vocal, color in VOWEL_COLORS.items():
            btn = tk.Button(
                frame_vocales,
                text=vocal,
                font=("Segoe UI", 22, "bold"),
                width=3,
                height=1,
                bg=color,
                fg="white",
                activebackground=self._oscurecer(color),
                activeforeground="white",
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda v=vocal: self._enviar(v.lower()),
            )
            btn.pack(side=tk.LEFT, padx=6)
            btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=self._aclarar(c)))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))

        # Separador
        tk.Frame(self.root, bg=ACCENT, height=1).pack(fill=tk.X, padx=30, pady=15)

        # Controles
        tk.Label(
            self.root,
            text="CONTROLES",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_DIM,
            bg=BG,
        ).pack(pady=(0, 5))

        frame_controles = tk.Frame(self.root, bg=BG)
        frame_controles.pack(pady=(0, 20))

        btn_reset = tk.Button(
            frame_controles,
            text="✋  Abrir mano",
            font=("Segoe UI", 13),
            width=14,
            bg="#2ecc71",
            fg="white",
            activebackground="#27ae60",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._enviar("reset"),
        )
        btn_reset.pack(side=tk.LEFT, padx=8)

        btn_cerrar = tk.Button(
            frame_controles,
            text="✊  Cerrar mano",
            font=("Segoe UI", 13),
            width=14,
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._enviar("cerrar"),
        )
        btn_cerrar.pack(side=tk.LEFT, padx=8)

        # Estado de último envío
        self.lbl_ultimo = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG,
        )
        self.lbl_ultimo.pack(pady=(0, 15))

    # ── Conexión ──

    def _conectar(self):
        ip = self.entry_ip.get().strip()
        uri = f"ws://{ip}:81"
        try:
            self.ws = websocket.create_connection(uri, timeout=3)
            self.conectado = True
            self.lbl_estado.config(text="● Conectado", fg="#2ecc71")
        except Exception:
            self.conectado = False
            self.lbl_estado.config(text="● Desconectado", fg="#e74c3c")

    def _reconectar(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.lbl_estado.config(text="● Conectando...", fg="#f5a623")
        self.root.update()
        threading.Thread(target=self._conectar, daemon=True).start()

    # ── Envío ──

    def _enviar(self, comando):
        if not self.conectado or not self.ws:
            messagebox.showwarning("Sin conexión", "No estás conectado a la Manito.\nVerifica la IP y presiona Conectar.")
            return
        try:
            self.ws.send(comando)
            nombres = {
                "a": "A  ✊👍", "e": "E  🤏", "i": "I  🤙",
                "o": "O  👌", "u": "U  ✌️",
                "reset": "Mano abierta ✋", "cerrar": "Puño cerrado ✊",
            }
            self.lbl_ultimo.config(text=f"Último: {nombres.get(comando, comando)}")
        except Exception as e:
            self.conectado = False
            self.lbl_estado.config(text="● Desconectado", fg="#e74c3c")
            messagebox.showerror("Error", f"Se perdió la conexión:\n{e}")

    # ── Utilidades de color ──

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
    app = ManitoApp(root)
    root.mainloop()
