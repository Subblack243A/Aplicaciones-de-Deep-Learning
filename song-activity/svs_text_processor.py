"""
SVSTextProcessor: Phonetic tokenization for Singing Voice Synthesis.
Converts text to integer sequences using an extended character alphabet.
"""

from __future__ import annotations


class SVSTextProcessor:
    """
    Tokenizes text for SVS models.
    Uses an extended alphabet with uppercase, lowercase, punctuation, and special tokens.
    """

    PAD = "_"
    EOS = "~"
    SPECIAL = [PAD, EOS]
    LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    PUNCTUATION = list("!\"'(),-.:;? ")

    SYMBOLS = SPECIAL + LETTERS + PUNCTUATION
    SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
    ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.SYMBOLS)

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self.SYMBOL_TO_ID[self.PAD]

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self.SYMBOL_TO_ID[self.EOS]

    def text_to_sequence(self, text: str) -> list[int]:
        """
        Converts text to a sequence of integer IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer IDs (with EOS appended).
        """
        sequence = []
        for char in text:
            if char in self.SYMBOL_TO_ID:
                sequence.append(self.SYMBOL_TO_ID[char])
        sequence.append(self.eos_id)
        return sequence

    def sequence_to_text(self, sequence: list[int]) -> str:
        """
        Converts a sequence of integer IDs back to text.

        Args:
            sequence: List of integer IDs.

        Returns:
            Decoded text string.
        """
        chars = []
        for idx in sequence:
            if idx in self.ID_TO_SYMBOL:
                symbol = self.ID_TO_SYMBOL[idx]
                if symbol not in (self.PAD, self.EOS):
                    chars.append(symbol)
        return "".join(chars)

    def get_sequence_length(self, text: str) -> int:
        """Returns the encoded length of a text (including EOS)."""
        return sum(1 for c in text if c in self.SYMBOL_TO_ID) + 1
