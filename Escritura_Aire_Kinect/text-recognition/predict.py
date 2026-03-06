import torch
from PIL import Image
from torchvision import transforms
try:
    from model import TextCNN
    from dataset import get_transforms
except ImportError:
    from .model import TextCNN
    from .dataset import get_transforms
import os

def predict_and_save(image_path, model_path=None):
    # 1. Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Si no se proporciona model_path, usar el de la misma carpeta del script
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
    
    # 2. Clases (debe coincidir con dataset.py)
    classes = ["duvan", "sierra", "felipe", "laura"]
    
    # 3. Cargar modelo
    model = TextCNN(num_classes=len(classes)).to(device)
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo de pesos {model_path}.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Preprocesar imagen
    transform = get_transforms()
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen {image_path}.")
        return
    
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device) # Batch dimension
    
    # 5. Inferencia
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = classes[predicted_idx.item()]
    
    print(f"Predicción: {predicted_class}")
    
    # 6. Guardar resultado en TXT
    # El archivo tendrá el nombre de la clase identificada
    txt_filename = "result.txt"
    txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), txt_filename)
    
    with open(txt_path, "w") as f:
        f.write(f"{predicted_class.upper()}")
    
    print(f"Resultado guardado en: {txt_path}")

if __name__ == "__main__":
    # Ejemplo de uso:
    # Asegúrate de tener una imagen para probar o pasa la ruta por parámetro
    # IMPORTANTE: El script espera que el archivo model.pth ya exista.
    test_image = os.path.join(os.path.dirname(__file__), "fel.jpeg")
    
    if os.path.exists(test_image):
        predict_and_save(test_image)
    else:
        print(f"Uso: Coloca una imagen llamada 'test.jpg' en esta carpeta o modifica el script.")
