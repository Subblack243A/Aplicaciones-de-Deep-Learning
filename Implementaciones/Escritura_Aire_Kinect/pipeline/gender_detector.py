"""
Gender Detector — Detección básica de género por nombre.
Usa una lista de nombres comunes en español + heurística por terminación.
"""

# Nombres femeninos comunes en español
_FEMALE_NAMES = {
    "laura", "maria", "ana", "carmen", "rosa", "pilar", "teresa", "isabel", "encarnacion",
    "francisca", "dolores", "juana", "cristina", "alicia", "marta", "luisa", "patricia",
    "sandra", "monica", "irene", "sara", "lucia", "elena", "nuria", "julia", "belen",
    "andrea", "alejandra", "vanessa", "natalia", "carolina", "paula", "claudia", "silvia",
    "inmaculada", "raquel", "eva", "marina", "gloria", "adriana", "victoria", "noelia",
    "aitana", "sofia", "valentina", "emma", "martina", "luciana", "camila", "renata",
}

# Nombres masculinos comunes en español
_MALE_NAMES = {
    "antonio", "jose", "manuel", "francisco", "juan", "david", "jesus", "carlos", "javier",
    "daniel", "miguel", "rafael", "pedro", "alejandro", "mario", "alberto", "sergio",
    "andres", "angel", "pablo", "fernando", "jorge", "luis", "diego", "eduardo", "ivan",
    "ruben", "raul", "enrique", "ramon", "vicente", "oscar", "joaquin", "marcos",
    "santiago", "nicolas", "sebastian", "mateo", "leonardo", "emiliano", "tomas",
    "agustin", "rodrigo", "guillermo", "martin", "hugo", "adrian", "alvaro", "mariano",
    "gonzalo", "ricardo", "felipe", "cristian", "marcelo", "dario", "federico", "julio",
    "esteban", "bruno", "maximiliano", "benjamin", "lucas", "thomas", "santos", "gabriel",
}


def detect_gender(name: str) -> str:
    """
    Recibe un nombre y devuelve 'female' o 'male'.
    Usa lista de nombres comunes + heurística por terminación.
    """
    name_clean = name.strip().lower().split()[0]  # Tomar primer nombre

    if name_clean in _FEMALE_NAMES:
        return "female"
    if name_clean in _MALE_NAMES:
        return "male"

    # Heurística por terminación (español)
    if name_clean.endswith(("a", "ia", "ina", "ra", "la", "sa", "ta", "ya", "za")):
        # Pero hay excepciones comunes masculinas terminadas en 'a'
        if name_clean in {"joshua", "noa", "noah", "elia", "ezekiela"}:
            return "male"
        return "female"

    if name_clean.endswith(("o", "io", "ro", "lo", "so", "to", "do", "no", "vo", "mo", "ño")):
        return "male"

    # Por defecto, si no sabemos, asumimos masculino (o se puede hacer configurable)
    return "male"
