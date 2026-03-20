import tkinter as tk
from tkinter import messagebox, scrolledtext
import websocket
import threading
import time

ESP32_IP = "192.168.1.50"

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
        self._escuchando = False

        self._crear_ui()
        self.root.after(100, self._reconectar)

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
        frame_controles.pack(pady=(0, 10))

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

        # Diagnóstico
        frame_diag = tk.Frame(self.root, bg=BG)
        frame_diag.pack(pady=(5, 0))

        btn_test = tk.Button(
            frame_diag,
            text="🔧 Test Pines",
            font=("Segoe UI", 11),
            width=30,
            bg="#f39c12",
            fg="white",
            activebackground="#e67e22",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._enviar("test"),
        )
        btn_test.pack()

        # Separador
        tk.Frame(self.root, bg=ACCENT, height=1).pack(fill=tk.X, padx=30, pady=10)

        # ── Serial Monitor inalámbrico ──
        tk.Label(
            self.root,
            text="SERIAL MONITOR (WiFi)",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_DIM,
            bg=BG,
        ).pack(pady=(0, 5))

        self.txt_log = scrolledtext.ScrolledText(
            self.root,
            font=("Consolas", 9),
            width=52,
            height=10,
            bg="#0d1117",
            fg="#58a6ff",
            insertbackground=TEXT,
            relief=tk.FLAT,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.txt_log.pack(padx=15, pady=(0, 5))

        btn_limpiar = tk.Button(
            self.root,
            text="Limpiar log",
            font=("Segoe UI", 9),
            bg=ACCENT,
            fg=TEXT_DIM,
            activebackground=HIGHLIGHT,
            activeforeground=TEXT,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._limpiar_log,
        )
        btn_limpiar.pack(pady=(0, 15))

    # ── Log ──

    def _agregar_log(self, texto):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, texto + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _log_seguro(self, texto):
        """Agrega al log desde cualquier thread de forma segura."""
        self.root.after(0, self._agregar_log, texto)

    def _limpiar_log(self):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.config(state=tk.DISABLED)

    # ── Conexión con escucha de mensajes ──

    def _conectar(self):
        ip = self.entry_ip.get().strip()
        uri = f"ws://{ip}:81"
        try:
            self.ws = websocket.create_connection(uri, timeout=3)
            self.conectado = True
            self.root.after(0, lambda: self.lbl_estado.config(text="● Conectado", fg="#2ecc71"))
            self._log_seguro(f"Conectado a {uri}")
            self._iniciar_escucha()
        except Exception as e:
            self.conectado = False
            self.root.after(0, lambda: self.lbl_estado.config(text="● Desconectado", fg="#e74c3c"))
            self._log_seguro(f"Error conectando: {e}")

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
        """Hilo que escucha mensajes del ESP32 (logs via WebSocket)."""
        self._escuchando = True

        def escuchar():
            while self._escuchando and self.ws:
                try:
                    self.ws.settimeout(1.0)
                    msg = self.ws.recv()
                    if msg:
                        self._log_seguro(f"ESP32> {msg}")
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:
                    if self._escuchando:
                        self.conectado = False
                        self._escuchando = False
                        self.root.after(0, lambda: self.lbl_estado.config(
                            text="● Desconectado", fg="#e74c3c"))
                        self._log_seguro("Conexion perdida.")
                    break

        threading.Thread(target=escuchar, daemon=True).start()

    # ── Envío ──

    def _enviar(self, comando):
        if not self.conectado or not self.ws:
            messagebox.showwarning(
                "Sin conexión",
                "No estás conectado a la Manito.\nVerifica la IP y presiona Conectar.",
            )
            return
        try:
            self.ws.send(comando)
            nombres = {
                "a": "A", "e": "E", "i": "I", "o": "O", "u": "U",
                "reset": "Abrir mano", "cerrar": "Cerrar mano",
                "test": "Test pines",
            }
            self._agregar_log(f">>> Enviado: {nombres.get(comando, comando)}")
        except Exception as e:
            self.conectado = False
            self._escuchando = False
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