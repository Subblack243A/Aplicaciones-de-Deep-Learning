# Generador Text to Image (T2I)

Entorno de entrenamiento híbrido de generación desde cero (Semilla Estocástica) enfocado a obtener letras vectorizadas / diagramas a partir de texto base, utilizado de forma nativa por el proyecto de "Manito Robótica".

### Funciones vitales
* **`descargar_dataset.py`**: Pre-configura un entorno en la PC para abstraer de Kaggle o repositorios estáticos los datasets visuales y sus labels asíncronamente en `.zip`.
* **`train_text2image.py`**: Implementa el codificador de atención `TransformerEncoder` con capas trans-convolucionales con una función de Loss perceptiva (LPIPS) para priorizar el realismo visual frente a la precisión per-pixel cuadrada tradicional de un MSE.

Al igual que en otras carpetas de entrenamiento, la interacción final se hace en la carpeta global *Implementaciones/Manito/*. Aquí solo residen parámetros de optimización.
