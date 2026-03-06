import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class TextDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes de texto escrito.
    Organización esperada: root_dir/clase/imagen.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["duvan", "sierra", "felipe", "laura"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Directory {cls_dir} not found.")
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L') # Convertir a escala de grises
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """
    Define las transformaciones obligatorias:
    - Redimensionar a 128x32.
    - Convertir a tensor.
    - Normalizar para 1 solo canal (Escala de grises).
    """
    return transforms.Compose([
        transforms.Resize((32, 128)), # Altura x Ancho
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalización simple para 1 canal
    ])

if __name__ == "__main__":
    # Prueba rápida de la lógica de transformación
    print("Mapeo de etiquetas:", {cls: i for i, cls in enumerate(["duvan", "sierra", "felipe", "laura"])})
    t = get_transforms()
    print("Transformaciones configuradas:", t)
