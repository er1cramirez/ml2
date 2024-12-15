import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usa la primera GPU
import sys
import torch
import torch.optim as optim
import tqdm
from torch import nn
from torchvision import transforms

# Obtener la ruta de src
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import utils.lfw_dataset_handler as lfw
import paths_config as paths

from model import Discriminator, Generator
# import src.paths_config as paths

def train_epoch(D, G, dataloader, d_optimizer, g_optimizer, device, k_t, gamma=0.5, lambda_k=0.001):
    """
    Entrena el modelo BEGAN por una época
    """
    D.train()
    G.train()
    d_epoch_loss = 0
    g_epoch_loss = 0
    
    for batch_idx, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Entrenar Discriminador
        d_optimizer.zero_grad()
        
        # Pérdida para imágenes reales
        real_reconstructed = D(real_images)
        d_real_loss = torch.mean(torch.abs(real_images - real_reconstructed))
        
        # Pérdida para imágenes falsas
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = G(noise)
        fake_reconstructed = D(fake_images.detach())
        d_fake_loss = torch.mean(torch.abs(fake_images.detach() - fake_reconstructed))
        
        # Pérdida total del discriminador
        d_loss = d_real_loss - k_t * d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Entrenar Generador
        g_optimizer.zero_grad()
        fake_reconstructed = D(fake_images)
        g_loss = torch.mean(torch.abs(fake_images - fake_reconstructed))
        g_loss.backward()
        g_optimizer.step()
        
        # Actualizar k_t usando max y min para limitar entre 0 y 1
        balance = (gamma * d_real_loss - d_fake_loss).item()
        k_t = k_t + lambda_k * balance
        k_t = max(0, min(1, k_t))  # Reemplazamos clamp con max/min
        
        # Medida de convergencia
        M = d_real_loss.item() + abs(balance)
        
        d_epoch_loss += d_loss.item()
        g_epoch_loss += g_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'[Batch {batch_idx}/{len(dataloader)}] '
                  f'D_loss: {d_loss.item():.4f} '
                  f'G_loss: {g_loss.item():.4f} '
                  f'M: {M:.4f} '
                  f'k_t: {k_t:.4f}')
        
    return d_epoch_loss/len(dataloader), g_epoch_loss/len(dataloader), k_t, M

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

def save_checkpoint(D, G, d_optimizer, g_optimizer, epoch, D_loss, G_loss, is_best=False, checkpoint_dir='./checkpoints'):
    """
    Guarda un checkpoint del modelo
    
    Args:
        D: Discriminador
        G: Generador
        d_optimizer: Optimizador del discriminador
        g_optimizer: Optimizador del generador
        epoch: Número de epoch
        D_loss: Pérdida del discriminador
        G_loss: Pérdida del generador
        is_best: Indica si es el mejor modelo
        checkpoint_dir: Directorio para guardar el checkpoint
    """
    D_checkpoint = {
        'epoch': epoch,
        'model_state_dict': D.state_dict(),
        'optimizer_state_dict': d_optimizer.state_dict(),
        'loss': D_loss
    }

    G_checkpoint = {
        'epoch': epoch,
        'model_state_dict': G.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict(),
        'loss': G_loss
    }

    D_checkpoint_path = f"{checkpoint_dir}/D_checkpoint_{epoch}.pth"
    G_checkpoint_path = f"{checkpoint_dir}/G_checkpoint_{epoch}.pth"

    torch.save(D_checkpoint, D_checkpoint_path)
    torch.save(G_checkpoint, G_checkpoint_path)

    if is_best:
        torch.save(D_checkpoint, f"{checkpoint_dir}/best_D_model.pth")
        torch.save(G_checkpoint, f"{checkpoint_dir}/best_G_model.pth")
        print(f"Saved best model at epoch {epoch}")
        # Clean up previous checkpoints if not best
        files = os.listdir(checkpoint_dir)
        for file in files:
            if file.startswith("D_checkpoint") and file != f"D_checkpoint_{epoch}.pth" and file != "best_D_model.pth":
                os.remove(f"{checkpoint_dir}/{file}")
            if file.startswith("G_checkpoint") and file != f"G_checkpoint_{epoch}.pth" and file != "best_G_model.pth":
                os.remove(f"{checkpoint_dir}/{file}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Carga un checkpoint del modelo
    
    Args:
        model: Modelo a cargar
        optimizer: Optimizador
        checkpoint_path: Ruta del checkpoint
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train(D, G, train_loader, val_loader, d_optimizer, g_optimizer, device, epochs=10, checkpoint_freq=5):
    """
    Entrena el modelo
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader con los datos de entrenamiento
        val_loader: DataLoader con los datos de validación
        criterion: Función de pérdida
        optimizer: Optimizador
        device: Dispositivo de cómputo
        epochs: Número de epochs
        checkpoint_freq: Frecuencia para guardar checkpoints
    """
    k_t = 0
    gamma = 0.5
    lambda_k = 0.001
    best_M = float('inf')

    if os.path.exists(paths.BEGAN_CHECKPOINT_DIR):
        # check if there is a best_checkpoint and load it if not, load the last checkpoint
        if os.path.exists(f"{paths.BEGAN_CHECKPOINT_DIR}/best_G_model.pth"):
            G, g_optimizer, start_epoch, best_G_loss = load_checkpoint(G, g_optimizer, f"{paths.BEGAN_CHECKPOINT_DIR}/best_G_model.pth")
            D, d_optimizer, _, best_D_loss = load_checkpoint(D, d_optimizer, f"{paths.BEGAN_CHECKPOINT_DIR}/best_D_model.pth")
            print(f"Loaded best model from epoch {start_epoch}")
        else:
            # load last checkpoint
            files = os.listdir(paths.BEGAN_CHECKPOINT_DIR)
            checkpoint_files = [file for file in files if file.startswith("G_checkpoint")]
            if not checkpoint_files:
                start_epoch = 0
            else:
                last_checkpoint = sorted(checkpoint_files)[-1]
                G, g_optimizer, start_epoch, _ = load_checkpoint(G, g_optimizer, f"{paths.BEGAN_CHECKPOINT_DIR}/{last_checkpoint}")
                D, d_optimizer, _, _ = load_checkpoint(D, d_optimizer, f"{paths.BEGAN_CHECKPOINT_DIR}/{last_checkpoint}")
                print(f"Loaded last model from epoch {start_epoch}")
    else:
        os.makedirs(paths.BEGAN_CHECKPOINT_DIR)
        start_epoch = 0
    
    for epoch in range(epochs):
        d_loss, g_loss, k_t, M = train_epoch(D, G, train_loader, d_optimizer, 
                                            g_optimizer, device, k_t, gamma, lambda_k)
        
        print(f'Epoch {epoch}/{epochs} '
              f'D_loss: {d_loss:.4f} '
              f'G_loss: {g_loss:.4f} '
              f'M: {M:.4f} '
              f'k_t: {k_t:.4f}')
        
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(D, G, d_optimizer, g_optimizer, epoch, d_loss, g_loss, is_best=False, checkpoint_dir=paths.BEGAN_CHECKPOINT_DIR)
        
        if M < best_M:
            best_M = M
            save_checkpoint(D, G, d_optimizer, g_optimizer, epoch, 
                          d_loss, g_loss, is_best=True, checkpoint_dir=paths.BEGAN_CHECKPOINT_DIR)
    
    # save final model
    save_checkpoint(D, G, d_optimizer, g_optimizer, epoch, d_loss, g_loss, is_best=False, checkpoint_dir=paths.BEGAN_MODEL_DIR)
    print("Training complete.")

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
    paths.setup_began_paths()
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

    # Ajustar learning rates según recomendaciones de BEGAN
    d_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # No necesitamos criterion ya que la pérdida está implementada en train_epoch
    train(D, G, train_loader, val_loader, d_optimizer, g_optimizer, device, epochs=100, checkpoint_freq=5)

if __name__ == '__main__':
    main()