import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """
    Arquitectura CNN ligera para clasificación de texto (4 clases).
    Entrada: (Batch, 1, 32, 128) -> Escala de grises, 128x32 píxeles.
    """
    def __init__(self, num_classes=4):
        super(TextCNN, self).__init__()
        
        # Bloque 1: Conv -> ReLU -> Pool
        # Entrada: [B, 1, 32, 128]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Después de conv1: [B, 32, 32, 128] (mantiene dimensión por padding=1)
        # Después de pool1: [B, 32, 16, 64]
        
        # Bloque 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Después de conv2: [B, 64, 16, 64]
        # Después de pool2: [B, 64, 8, 32]
        
        # Bloque 3: Conv -> ReLU -> Pool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Después de conv3: [B, 128, 8, 32]
        # Después de pool3: [B, 128, 4, 16]
        
        # Capas Densas (Fully Connected)
        # El mapa de características final es 128 * 4 * 16
        self.fc1 = nn.Linear(128 * 4 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [Batch, 1, 32, 128]
        x = self.pool1(F.relu(self.conv1(x))) # -> [B, 32, 16, 64]
        x = self.pool2(F.relu(self.conv2(x))) # -> [B, 64, 8, 32]
        x = self.pool3(F.relu(self.conv3(x))) # -> [B, 128, 4, 16]
        
        # Aplanar el tensor para la capa densa
        x = x.view(-1, 128 * 4 * 16)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Salida de logits (sin softmax para CrossEntropyLoss)
        
        return x

if __name__ == "__main__":
    # Prueba de dimensiones
    model = TextCNN(num_classes=4)
    dummy_input = torch.randn(1, 1, 32, 128) # Solo 1 canal (Grarscale)
    output = model(dummy_input)
    print(f"Arquitectura del modelo:\n{model}")
    print(f"\nForma de entrada: {dummy_input.shape}")
    print(f"Forma de salida: {output.shape} (Debe ser [1, 4])")
