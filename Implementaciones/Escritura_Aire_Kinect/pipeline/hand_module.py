"""
Hand Module — Comunicación con la mano robótica (Manito).
Soporta modo mock (simulado) y modo websocket (real).

Variable de entorno HAND_MODE=mock|websocket
"""
import os
import json
import threading
import time

from . import config


class HandController:
    """Controlador de la mano robótica."""

    def __init__(self):
        self.mode = config.HAND_MODE
        self.ws = None
        self.connected = False
        self._letters_queue = []
        self._queue_active = False
        self._queue_cancelled = False

    # ── Conexión ──────────────────────────────────────────────────────

    def connect(self):
        if self.mode == "mock":
            print("[Hand] Modo MOCK activado. No se conectará a ningún dispositivo.")
            self.connected = True
            return True

        try:
            import websocket
        except ImportError:
            print("[Hand] ERROR: websocket-client no está instalado. pip install websocket-client")
            self.connected = False
            return False

        url = f"ws://{config.HAND_IP}:{config.HAND_PORT}/ws"
        try:
            self.ws = websocket.create_connection(url, timeout=3)
            self.connected = True
            print(f"[Hand] Conectado a {url}")
            return True
        except Exception as e:
            print(f"[Hand] Error conectando a {url}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        self.connected = False

    # ── Envío de comandos ─────────────────────────────────────────────

    def send_command(self, command: str):
        """Envía un comando crudo a la mano."""
        if self.mode == "mock":
            print(f"[Hand][MOCK] Comando enviado: {command}")
            return True

        if not self.connected or self.ws is None:
            print("[Hand] No conectado. Intentando reconectar...")
            if not self.connect():
                return False

        try:
            self.ws.send(command)
            return True
        except Exception as e:
            print(f"[Hand] Error enviando comando: {e}")
            self.connected = False
            return False

    def send_letter(self, letter: str):
        """Envía una letra individual al abecedario LSC."""
        letter = letter.lower().strip()
        if not letter:
            return False
        return self.send_command(letter)

    def send_text(self, text: str):
        """
        Envía un texto completo letra por letra.
        Filtra solo letras válidas del abecedario LSC (A-Z + Ñ).
        """
        valid_letters = set("abcdefghijklmnopqrstuvwxyzñ")
        letters = [ch for ch in text.lower() if ch in valid_letters]

        if not letters:
            print("[Hand] No hay letras válidas para enviar.")
            return

        print(f"[Hand] Enviando {len(letters)} letras: {' '.join(letters).upper()}")

        def _worker():
            for ch in letters:
                if self._queue_cancelled:
                    print("[Hand] Cola cancelada.")
                    self._queue_cancelled = False
                    self._queue_active = False
                    return
                self.send_letter(ch)
                time.sleep(config.HAND_DELAY_BETWEEN_LETTERS_MS / 1000.0)
            self._queue_active = False
            print("[Hand] Todas las letras enviadas.")

        self._queue_cancelled = False
        self._queue_active = True
        threading.Thread(target=_worker, daemon=True).start()

    def cancel(self):
        """Cancela el envío actual."""
        self._queue_cancelled = True

    def reset_hand(self):
        """Envía comando para abrir/resetear la mano."""
        self.send_command("reset")

    def close_hand(self):
        """Envía comando para cerrar la mano."""
        self.send_command("cerrar")
