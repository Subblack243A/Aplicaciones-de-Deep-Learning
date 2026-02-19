"""
TextEncoder: Codificación y decodificación de texto a secuencias de enteros.
Define el vocabulario (caracteres a-z, espacio, apóstrofe, paréntesis) y el token blank para CTC.
"""


class TextEncoder:
    """Codifica texto a enteros y viceversa para el modelo ASR."""

    # Vocabulario: letras minúsculas + espacio + caracteres especiales
    VOCAB = list("abcdefghijklmnopqrstuvwxyz '")
    CHAR_TO_INT = {char: i for i, char in enumerate(VOCAB)}
    INT_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}
    BLANK_TOKEN = len(VOCAB)  # Token CTC blank = último índice + 1

    @property
    def vocab_size(self) -> int:
        """Tamaño del vocabulario (sin contar blank token)."""
        return len(self.VOCAB)

    @property
    def total_tokens(self) -> int:
        """Tamaño total de tokens incluyendo blank token para CTC."""
        return len(self.VOCAB) + 1

    def encode(self, text: str) -> list[int]:
        """
        Convierte texto a una lista de enteros.

        Args:
            text: Texto a codificar (se convierte a minúsculas).

        Returns:
            Lista de enteros correspondientes a cada carácter.
        """
        text = text.lower().strip()
        encoded = []
        for char in text:
            if char in self.CHAR_TO_INT:
                encoded.append(self.CHAR_TO_INT[char])
            # Ignorar caracteres no reconocidos
        return encoded

    def decode(self, ints: list[int]) -> str:
        """
        Convierte una lista de enteros a texto.

        Args:
            ints: Lista de enteros.

        Returns:
            Texto decodificado.
        """
        chars = []
        for i in ints:
            if i in self.INT_TO_CHAR:
                chars.append(self.INT_TO_CHAR[i])
            # Ignorar blank tokens y tokens desconocidos
        return "".join(chars)

    def decode_ctc_output(self, ints: list[int]) -> str:
        """
        Decodifica la salida CTC eliminando repeticiones y blanks.

        Args:
            ints: Lista de enteros (salida CTC).

        Returns:
            Texto decodificado sin repeticiones ni blanks.
        """
        chars = []
        prev = None
        for i in ints:
            if i == self.BLANK_TOKEN:
                prev = i
                continue
            if i != prev and i in self.INT_TO_CHAR:
                chars.append(self.INT_TO_CHAR[i])
            prev = i
        return "".join(chars)
