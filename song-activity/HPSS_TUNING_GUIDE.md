# Guía de Ajuste de HPSS (Separación Armónica-Percusiva)

Esta guía explica cómo modificar los parámetros del algoritmo HPSS en `main.py` para obtener diferentes resultados en la separación de audio (Voz vs Música).

## Parámetros Principales

La función `hpss_manual` acepta los siguientes argumentos:

### 1. `margin_harmonic` (Margen Armónico)

Controla qué tan "fuerte" debe ser un sonido armónico (voz/instrumentos melódicos) para ser capturado.

* **Mayor a 1.0 (ej. 3.0)**: **Agresivo**. Captura mucha señal como "Armónica".
  * *Efecto*: La voz suena más completa, pero se filtran más instrumentos (piano, guitarras).
  * *Uso*: Cuando quieres asegurar que no se pierda nada de la voz.
* **Menor a 1.0 (ej. 0.5)**: **Estricto**. Solo captura lo puramente horizontal/armónico.
  * *Efecto*: La voz queda muy limpia pero puede sonar "rota" o metálica. Elimina casi toda la música.
  * *Uso*: Para análisis de voz muy específico.

### 2. `margin_percussive` (Margen Percusivo)

Controla la captura de golpes, batería y transitorios.

* **Mayor a 1.0 (ej. 3.0)**: Captura más ruido y batería.
* **Menor a 1.0 (ej. 0.5)**: Solo captura los golpes más fuertes de la batería.

### 3. `kernel_size` (Tamaño del Filtro)

Define el tamaño de la ventana de análisis (en píxeles del espectrograma).

* **31 (Default)**: Balanceado.
* **15 (Pequeño)**: Mejor para voces rápidas o cambios rápidos, pero separa peor.
* **61+ (Grande)**: Mejor separación en notas largas sostenidas, pero puede difuminar detalles rápidos.

---

## Estrategias de "Tuning" (Ajuste)

Para modificar estos valores, edita la línea en `main.py`:

```python
y_voice, y_drums, y_music = hpss_manual(y, sr, margin_harmonic=2.5, margin_percussive=1.0, kernel_size=31)
```

### Escenario A: "Quiero la voz lo más limpia posible" (Vocal Isolation Idea)

Priorizamos que solo pase lo armónico fuerte, sacrificando cuerpo de la voz.

* `margin_harmonic` = **0.8** a **1.5**
* `margin_percussive` = **2.0** (Para que todo lo dudoso se vaya al fondo/percusión)

### Escenario B: "Quiero una pista de Karaoke (Música sin voz)"

Necesitamos que la "Voz" capture TODO lo que parezca voz, para poder restarlo bien de la mezcla original.

* `margin_harmonic` = **3.0** a **4.0** (Capturamos toda la voz y un poco más)
* `margin_percussive` = **1.0**
* **Resultado**: Al restar esta "Super Voz" de la mezcla, la pista de música (`_music_no_voice.wav`) quedará con menos residuos vocales.

### Escenario C: "Quiero separar solo la Batería"

* `margin_harmonic` = **1.0**
* `margin_percussive` = **3.0** (Prioridad a lo vertical)
