import os
import sys

# Simular la ruta absoluta
current_dir = os.path.dirname(os.path.abspath(__file__))
text_recognition_path = os.path.join(current_dir, "text-recognition")

print(f"Buscando 'predict.py' en: {text_recognition_path}")
sys.path.append(text_recognition_path)

try:
    from predict import predict_and_save
    print("Éxito: Se pudo importar 'predict_and_save' correctamente.")
except ImportError as e:
    print(f"Error: No se pudo importar 'predict_and_save': {e}")
    sys.exit(1)

# Probar con una imagen de ejemplo si existe
example_image = os.path.join(text_recognition_path, "fel.jpeg")
if os.path.exists(example_image):
    print(f"Probando predicción con: {example_image}")
    try:
        predict_and_save(example_image)
        print("Éxito: La función 'predict_and_save' se ejecutó sin errores.")
        
        # Verificar si se actualizó result.txt
        result_path = os.path.join(text_recognition_path, "result.txt")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                content = f.read()
            print(f"Contenido de 'result.txt': {content}")
        else:
            print("Aviso: 'result.txt' no se encontró tras la predicción.")
            
    except Exception as e:
        print(f"Error durante la ejecución de 'predict_and_save': {e}")
else:
    print(f"Aviso: No se encontró la imagen de prueba {example_image}. No se puede probar la ejecución completa.")
