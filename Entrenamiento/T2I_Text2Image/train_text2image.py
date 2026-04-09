import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import matplotlib.pyplot as plt
import numpy as np
import os
from captum.attr import LayerIntegratedGradients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class EarlyStopping:
    """Implementa el Early Stopping basado en val_loss o límite de épocas."""
    def __init__(self, patience=10, min_delta=0.001, max_epochs=200):
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epochs_run = 0

    def __call__(self, val_loss):
        self.epochs_run += 1
        
        if self.epochs_run >= self.max_epochs:
            print("Límite máximo de épocas alcanzado. Deteniendo entrenamiento.")
            self.early_stop = True
            return

        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping activado. Paciencia de {self.patience} agotada.")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        embedded = self.embedding(x)
        features = self.transformer(embedded)
        return features

class ImageDecoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.fc_noise = nn.Linear(128, 16 * 16 * d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.upconv1 = nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.upconv2 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise, text_features):
        batch_size = noise.size(0)
        x = self.fc_noise(noise).view(batch_size, -1, 256)
        attn_out, _ = self.cross_attn(query=x, key=text_features, value=text_features)
        x = x + attn_out
        x = x.permute(0, 2, 1).view(batch_size, 256, 16, 16)
        x = self.relu(self.upconv1(x))
        img_out = self.tanh(self.upconv2(x))
        return img_out

class Text2ImageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size)
        self.image_decoder = ImageDecoder()

    def forward(self, text, noise):
        text_features = self.text_encoder(text)
        img = self.image_decoder(noise, text_features)
        return img


def train_model(model, dataloader, epochs, vocab_size):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    mse_loss_fn = nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    lpips_metric = lpips.LPIPS(net='vgg').to(device) 
    
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, max_epochs=200)

    history = {'loss': [], 'ssim': [], 'lpips': []}

    print("Iniciando entrenamiento...")
    for epoch in range(epochs):
        model.train()
        total_loss, total_ssim, total_lpips = 0, 0, 0
        
        for text, real_images in dataloader:
            text, real_images = text.to(device), real_images.to(device)
            noise = torch.randn(text.size(0), 128).to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                fake_images = model(text, noise)
                
                loss_mse = mse_loss_fn(fake_images, real_images)
                loss_lpips = lpips_metric(fake_images, real_images).mean()
                
                loss = loss_mse + (0.1 * loss_lpips) 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.empty_cache()

            total_loss += loss.item()
            with torch.no_grad():
                total_ssim += ssim_metric(fake_images, real_images).item()
                total_lpips += loss_lpips.item()

        avg_loss = total_loss / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        avg_lpips = total_lpips / len(dataloader)
        
        history['loss'].append(avg_loss)
        history['ssim'].append(avg_ssim)
        history['lpips'].append(avg_lpips)
        
        print(f"Época {epoch+1} | Loss: {avg_loss:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
        
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            break

    return history


def plot_metrics(history):
    """Genera gráficas de evolución de pérdida y métricas."""
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], label='Loss (MSE+LPIPS)', color='red')
    plt.title('Evolución de Loss')
    plt.xlabel('Épocas')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['ssim'], label='SSIM', color='blue')
    plt.title('Similitud Estructural (Higher is better)')
    plt.xlabel('Épocas')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['lpips'], label='LPIPS', color='green')
    plt.title('Pérdida Perceptual (Lower is better)')
    plt.xlabel('Épocas')
    plt.legend()

    plt.tight_layout()
    plt.savefig("metricas_entrenamiento.png")
    print("Gráfica guardada como 'metricas_entrenamiento.png'")

def explain_attention_with_captum(model, text_tensor, noise_tensor):
    """
    Usa LayerIntegratedGradients de Captum para ver qué tokens del texto 
    tuvieron más impacto en la generación de la imagen.
    """
    model.eval()
    
    lig = LayerIntegratedGradients(model, model.text_encoder.embedding)
    
    def custom_forward(text_inputs):
        return model(text_inputs, noise_tensor).sum(dim=(1, 2, 3))
    
    attributions, delta = lig.attribute(inputs=text_tensor,
                                        target=None,
                                        additional_forward_args=(),
                                        custom_attribution_func=None,
                                        return_convergence_delta=True)
    
    word_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    
    print("\n--- Mapas de Atención (Importancia por Token) ---")
    for i, attr in enumerate(word_attributions):
        print(f"Token ID {text_tensor[0][i].item()}: Importancia = {attr:.4f}")
    
    return word_attributions