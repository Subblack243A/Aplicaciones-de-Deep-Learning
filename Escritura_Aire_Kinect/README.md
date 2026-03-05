# Escritura en el Aire con Kinect v1 (Xbox 360)

Este proyecto es una aplicación de visión artificial y Deep Learning que permite escribir en el aire utilizando un sensor Kinect modelo 1414. Combina el rastreo de articulaciones de la mano, reconocimiento de gestos y digitalización de texto (OCR).

## El Desafío Técnico: Porting del Wrapper

El mayor desafío técnico de este proyecto fue la obsolescencia del hardware. El Kinect v1 depende del Kinect for Windows SDK v1.8, el cual solo cuenta con soporte oficial para Python 2.7.

Para solucionar esto, se desarrolló un port nativo (Wrapper) a medida para Python 3.10. Se identificó que el SDK utiliza la librería Kinect10.dll, por lo que se reconstruyeron los módulos structs.py y \_interop.py utilizando la librería ctypes para mapear las tablas virtuales (vtable) de la interfaz COM del Kinect directamente a la versión actual de Python. Se implementó un sistema de multithreading para capturar el buffer de video BGRA del Kinect y sincronizarlo con el ciclo de procesamiento sin latencia. Esto elimina la necesidad de usar herramientas externas como OBS o cámaras virtuales, logrando una conexión directa y eficiente.

## Librerias utilizadas

MediaPipe fue utilizado para el seguimiento de los 21 puntos de referencia de la mano y detección de gestos. La librería permite obtener coordenadas precisas en tiempo real para el control del puntero.

EasyOCR apoyado en PyTorch se empleó como el motor de reconocimiento. Utiliza una red neuronal basada en arquitecturas CRNN y CTC para digitalizar los trazos dibujados y convertirlos a texto plano.

OpenCV se encargó del procesamiento de imagen, incluyendo el espejado del video y el renderizado de la interfaz de usuario directamente sobre el flujo de la cámara.

Tkinter se utilizó para construir el panel de control lateral, permitiendo al usuario ajustar el grosor del trazo y gestionar la exportación de archivos PNG y TXT.

MobileNetV2 integrado en PyTorch se incluyó como la arquitectura base para la clasificación avanzada de gestos mediante técnicas de Transfer Learning.

## Estructura del Módulo pykinect_v1

| Archivo          | Función                                                                                          |
| :--------------- | :----------------------------------------------------------------------------------------------- |
| nui/structs.py   | Definición de estructuras de datos C++ (Vectors, ImageFrames) portadas a Python mediante Ctypes. |
| nui/\_interop.py | Puente de bajo nivel con las funciones exportadas de la librería Kinect10.dll.                   |
| nui/**init**.py  | Capa de abstracción de alto nivel para gestionar el Runtime, la Cámara y los flujos de datos.    |

## Funcionalidades y Gestos

1. Escritura: Se activa al levantar únicamente el dedo índice. La aplicación detecta el punto exacto de la punta del dedo para trazar sobre el lienzo digital.

2. Pausa: Se activa al cerrar el puño. El sistema detiene el trazo permitiendo el movimiento de la mano sin generar dibujos accidentales.

3. Selector de Color: Permite cambiar entre Rojo, Verde y Azul tocando botones virtuales situados en la parte superior del flujo de video.

4. Digitalización: Al terminar, el sistema procesa el lienzo mediante el lector de caracteres para convertir los trazos en un archivo de texto editable.

---

Proyecto para la asignatura: Aplicaciones de Deep Learning  
Universidad de Cundinamarca - Semestre IX
