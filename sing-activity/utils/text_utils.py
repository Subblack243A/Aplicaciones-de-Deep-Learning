import re
import numpy as np

# ============================================================
# MÓDULO 1: Tokenización fonética simplificada (sin modelos externos)
# ============================================================

# Alfabeto fonético reducido (ARPAbet simplificado)
_pad = '_'
_eos = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz¡!\'\"(),-.:;¿? '

# Diccionario de símbolos
symbols = [_pad, _eos] + list(_characters)
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text: str) -> list:
    """Convierte texto limpio a secuencia de IDs numéricos."""
    text = text.strip()
    sequence = []
    for char in text:
        if char in symbol_to_id:
            sequence.append(symbol_to_id[char])
    sequence.append(symbol_to_id[_eos])
    return sequence

def sequence_to_text(sequence: list) -> str:
    """Convierte secuencia de IDs de vuelta a texto."""
    return ''.join(id_to_symbol.get(s, '') for s in sequence)
