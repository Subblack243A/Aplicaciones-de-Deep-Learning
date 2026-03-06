import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextDataset, get_transforms
from model import TextCNN
import os

def train_model(data_dir, num_epochs=20, batch_size=16, learning_rate=0.001):
    # 1. Configurar dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 2. Cargar datos
    transform = get_transforms()
    dataset = TextDataset(root_dir=data_dir, transform=transform)
    
    if len(dataset) == 0:
        print(f"Error: No se encontraron imágenes en {data_dir}. Asegúrate de que las subcarpetas 'duvan', 'david', 'felipe' y 'laura' existan y contengan imágenes.")
        return

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Inicializar modelo, pérdida y optimizador
    model = TextCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Bucle de entrenamiento
    print("Iniciando entrenamiento...")
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Cálculo de precisión
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Época [{epoch+1}/{num_epochs}] - Pérdida: {epoch_loss:.4f} - Precisión: {epoch_acc:.2f}%")

    # 5. Guardar el modelo
    save_path = os.path.join(os.path.dirname(__file__), "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Entrenamiento completado. Modelo guardado en: {save_path}")

if __name__ == "__main__":
    # Suponiendo que los datos están en una carpeta llamada 'data' dentro de 'text-recognition'
    # o donde el usuario los tenga organizados.
    data_path = os.path.join(os.path.dirname(__file__), "data")
    
    # Crear carpeta de datos si no existe (solo para estructura demostrativa)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        for cls in ["duvan", "david", "felipe", "laura"]:
            os.makedirs(os.path.join(data_path, cls), exist_ok=True)
        print(f"Carpeta de datos creada en {data_path}. Por favor, coloca las imágenes en las subcarpetas correspondientes.")
    
    # Ejecutar entrenamiento
    # Se reduce el número de épocas para la prueba inicial si el usuario lo ejecuta sin datos
    train_model(data_dir=data_path, num_epochs=10)
