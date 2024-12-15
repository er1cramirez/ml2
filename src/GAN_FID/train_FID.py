import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usa la primera GPU
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import numpy as np
from scipy import linalg
from torchvision import transforms

# Obtener la ruta de src
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import utils.lfw_dataset_handler as lfw
import paths_config as paths

from model import Discriminator, Generator
# import src.paths_config as paths


from torch.utils.data import DataLoader

class InceptionV3Features(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        # Usar hasta la última capa pool
        self.model = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2
        ).eval()
        
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

def calculate_fid(D, G, train_loader, inception_model, device, n_samples=2000, batch_size=32):
    """
    Calcula el FID score entre imágenes reales y generadas de manera optimizada para memoria
    """
    inception_model.eval()
    G.eval()
    
    def get_features(images, model):
        """Extrae características usando el modelo inception"""
        images = nn.functional.interpolate(images, size=(299, 299), 
                                         mode='bilinear', align_corners=False)
        features = model(images)
        return features.view(features.size(0), -1)
    
    def calculate_statistics(features):
        """Calcula media y covarianza de manera eficiente"""
        mu = np.mean(features, axis=0)
        # Calcular covarianza en chunks para evitar problemas de memoria
        n_chunks = 10
        chunk_size = features.shape[0] // n_chunks
        cov_sum = np.zeros((features.shape[1], features.shape[1]))
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_chunks - 1 else features.shape[0]
            chunk = features[start_idx:end_idx]
            chunk_centered = chunk - mu
            cov_sum += chunk_centered.T @ chunk_centered
        
        sigma = cov_sum / (features.shape[0] - 1)
        return mu, sigma
    
    # Procesar imágenes reales en batches más pequeños
    real_features_list = []
    n_processed = 0
    
    with torch.no_grad():
        for real_imgs, _ in train_loader:
            if n_processed >= n_samples:
                break
                
            current_batch_size = min(batch_size, n_samples - n_processed)
            if real_imgs.size(0) > current_batch_size:
                real_imgs = real_imgs[:current_batch_size]
                
            real_imgs = real_imgs.to(device)
            features = get_features(real_imgs, inception_model)
            real_features_list.append(features.cpu().numpy())
            
            n_processed += real_imgs.size(0)
            
    real_features = np.concatenate(real_features_list, axis=0)[:n_samples]
    
    # Procesar imágenes generadas
    fake_features_list = []
    n_processed = 0
    
    with torch.no_grad():
        while n_processed < n_samples:
            current_batch_size = min(batch_size, n_samples - n_processed)
            noise = torch.randn(current_batch_size, G.latent_dim, device=device)
            fake_imgs = G(noise)
            features = get_features(fake_imgs, inception_model)
            fake_features_list.append(features.cpu().numpy())
            
            n_processed += current_batch_size
            
    fake_features = np.concatenate(fake_features_list, axis=0)[:n_samples]
    
    # Calcular estadísticas
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_fake, sigma_fake = calculate_statistics(fake_features)
    
    # Calcular FID
    diff = mu_real - mu_fake
    
    # Calcular la raíz cuadrada de la matriz de manera más eficiente
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid_score)

def train_epoch(D, G, dataloader, criterion, d_optimizer, g_optimizer, device):
    """
    Entrena el modelo por un epoch
    
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con los datos de entrenamiento
        criterion: Función de pérdida
        optimizer: Optimizador
        device: Dispositivo de cómputo
    
    Returns:
        float: Pérdida promedio del epoch
    """
    D.train()
    G.train()
    d_epoch_loss = 0
    g_epoch_loss = 0
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Labels para el entrenamiento
        real_label = torch.ones(batch_size, device=device)
        fake_label = torch.zeros(batch_size, device=device)
        
        # Entrenar Discriminador
        d_optimizer.zero_grad()
        output_real = D(real_images)
        d_loss_real = criterion(output_real, real_label)
        
        noise = torch.randn(batch_size, 100, device=device)
        fake_images = G(noise)
        output_fake = D(fake_images.detach())
        d_loss_fake = criterion(output_fake, fake_label)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Entrenar Generador
        g_optimizer.zero_grad()
        output_fake = D(fake_images)
        g_loss = criterion(output_fake, real_label)
        g_loss.backward()
        g_optimizer.step()
        
        # Actualizar pérdidas
        d_epoch_loss += d_loss.item()
        g_epoch_loss += g_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'[Batch {batch_idx}/{len(dataloader)}] '
                    f'D_loss: {d_loss.item():.4f} '
                    f'G_loss: {g_loss.item():.4f}')
        
    return d_epoch_loss / len(dataloader), g_epoch_loss / len(dataloader)

def validate(D, dataloader, criterion, device):
    """
    Evalúa el modelo en el conjunto de validación
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con los datos de validación
        criterion: Función de pérdida
        device: Dispositivo de cómputo

    Returns:
        float: Pérdida promedio en el conjunto de validación
    """
    # Calcular pérdida de validación
    val_loss = 0
    D.eval()
    with torch.no_grad():
        for val_images, _ in dataloader:
            val_images = val_images.to(device)
            val_output = D(val_images)
            val_loss += criterion(val_output, torch.ones_like(val_output)).item()
    val_loss /= len(dataloader)
    return val_loss

def train(D, G, train_loader, val_loader, criterion, d_optimizer, g_optimizer, 
          device, epochs=50, checkpoint_freq=5, fid_freq=5, fid_n_samples=1000, fid_batch_size=32):
    """
    Versión actualizada de la función de entrenamiento con parámetros FID optimizados
    """
    inception_model = InceptionV3Features().to(device)
    best_fid = float('inf')
    
    if os.path.exists(paths.GAN_FID_CHECKPOINT_DIR):
        if os.path.exists(f"{paths.GAN_FID_CHECKPOINT_DIR}/best_G_model.pth"):
            D, G, d_optimizer, g_optimizer, start_epoch, _, _, best_fid = load_checkpoint(
            D, G, d_optimizer, g_optimizer,
            f"{paths.GAN_FID_CHECKPOINT_DIR}/best_model.pth"
        )
            print(f"Loaded best model from epoch {start_epoch}")
        else:
            files = os.listdir(paths.GAN_FID_CHECKPOINT_DIR)
            checkpoint_files = [f for f in files if f.startswith("G_checkpoint")]
            if checkpoint_files:
                last_checkpoint = sorted(checkpoint_files)[-1]
                D, G, d_optimizer, g_optimizer, start_epoch, _, _, best_fid = load_checkpoint(
                D, G, d_optimizer, g_optimizer,
                f"{paths.GAN_CHECKPOINT_DIR}/{last_checkpoint}"
            )
                print(f"Loaded last model from epoch {start_epoch}")
            else:
                start_epoch = 0
    else:
        os.makedirs(paths.GAN_FID_CHECKPOINT_DIR)
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        # Entrenar por una época
        d_loss, g_loss = train_epoch(D, G, train_loader, criterion, 
                                   d_optimizer, g_optimizer, device)
        val_loss = validate(D, val_loader, criterion, device)
        
        print(f'Epoch {epoch}/{epochs} '
              f'D_loss: {d_loss:.4f} '
              f'G_loss: {g_loss:.4f} '
              f'Val_loss: {val_loss:.4f}')
        
        # Calcular FID con parámetros optimizados
        if (epoch + 1) % fid_freq == 0 or epoch == epochs - 1:
            current_fid = calculate_fid(D, G, train_loader, inception_model, device,
                                      n_samples=fid_n_samples, batch_size=fid_batch_size)
            print(f'FID Score: {current_fid:.2f}')
            
            # Guardar si es el mejor modelo según FID
            if current_fid < best_fid:
                best_fid = current_fid
                save_checkpoint(D, G, d_optimizer, g_optimizer, epoch, d_loss, g_loss, 
                              current_fid, is_best=True, checkpoint_dir=paths.GAN_FID_CHECKPOINT_DIR)
                print(f'Nuevo mejor FID: {best_fid:.2f}')
        
        # Guardar checkpoint regular
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(D, G, d_optimizer, g_optimizer, epoch, d_loss, g_loss, 
                          current_fid if 'current_fid' in locals() else None, 
                          is_best=False, checkpoint_dir=paths.GAN_FID_CHECKPOINT_DIR)

    # Guardar modelo final
    save_checkpoint(D, G, d_optimizer, g_optimizer, epochs-1, d_loss, g_loss,
                   current_fid if 'current_fid' in locals() else None,
                   is_best=False, checkpoint_dir=paths.GAN_FID_MODEL_DIR)
    print("Training complete. Best FID:", best_fid)

def save_checkpoint(D, G, d_optimizer, g_optimizer, epoch, D_loss, G_loss, fid_score=None,
                   is_best=False, checkpoint_dir='./checkpoints'):
    """
    Versión actualizada de save_checkpoint que incluye FID score
    """
    checkpoint = {
        'epoch': epoch,
        'D_state_dict': D.state_dict(),
        'G_state_dict': G.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'D_loss': D_loss,
        'G_loss': G_loss,
        'fid_score': fid_score
    }

    # Guardar checkpoint regular
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)

    # Guardar mejor modelo si aplica
    if is_best:
        best_path = f"{checkpoint_dir}/best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model with FID: {fid_score:.2f}")
        
        # Limpiar checkpoints anteriores
        files = os.listdir(checkpoint_dir)
        for file in files:
            if file.startswith("checkpoint_epoch") and file != f"checkpoint_epoch_{epoch}.pth":
                os.remove(f"{checkpoint_dir}/{file}")


def load_checkpoint(D, G, d_optimizer, g_optimizer, checkpoint_path):
    """
    Carga un checkpoint que contiene tanto el Discriminador como el Generador
    
    Args:
        D: Modelo Discriminador
        G: Modelo Generador
        d_optimizer: Optimizador del discriminador
        g_optimizer: Optimizador del generador
        checkpoint_path: Ruta del checkpoint
    
    Returns:
        tuple: (D, G, d_optimizer, g_optimizer, epoch, d_loss, g_loss, fid_score)
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Cargar estados de los modelos
    D.load_state_dict(checkpoint['D_state_dict'])
    G.load_state_dict(checkpoint['G_state_dict'])
    
    # Cargar estados de los optimizadores
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    
    # Obtener resto de información
    epoch = checkpoint['epoch']
    d_loss = checkpoint['D_loss']
    g_loss = checkpoint['G_loss']
    fid_score = checkpoint.get('fid_score', None)  # Usar .get para manejar checkpoints antiguos
    
    print(f"Loaded checkpoint from epoch {epoch}")
    if fid_score is not None:
        print(f"FID score: {fid_score:.2f}")
    
    return D, G, d_optimizer, g_optimizer, epoch, d_loss, g_loss, fid_score


def get_device():
    if torch.cuda.is_available():
        try:
            # Verifica que CUDA realmente funcione
            test_tensor = torch.tensor([1.0]).cuda()
            return torch.device('cuda:0')  # Especifica explícitamente cuda:0
        except RuntimeError as e:
            print(f"CUDA test failed: {e}")
            return torch.device('cpu')
    return torch.device('cpu')
    
def main():
    # Configurar ruta de datos
    paths.setup_gan_fid_paths()
    # Configurar dispositivo
    device = get_device()
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Definir transformaciones
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Obtener dataloaders
    train_loader, val_loader, test_loader = lfw.get_data_loaders(
        batch_size=64,
        base_dir=paths.BASE_DATA_DIR,
        download=False,
        transform=transform
    )

    # Inicializar modelos
    D = Discriminator().to(device)
    G = Generator().to(device)

    # Inicializar optimizadores y función de pérdida    
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # fixed_noise = torch.randn(16, latent_dim=100, device=device)

    # Entrenar modelo
    train(D, G, train_loader, val_loader, criterion, d_optimizer, g_optimizer, device, epochs=15)

if __name__ == '__main__':
    main()