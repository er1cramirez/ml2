import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usa la primera GPU
import sys
import torch
import tqdm
from torch import nn
from torchvision import transforms

# Obtener la ruta de src
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import utils.lfw_dataset_handler as lfw
import paths_config as paths
# import src.paths_config as paths


# Definir modelo
class ConvAutoencoder(nn.Module):
    """
    Autoencoder Convolucional para Rostros
    =====================================
    
    Arquitectura del modelo:
    
    Encoder:
    --------
    1. Input (3, 128, 128) -> Conv2d -> (32, 64, 64)   [Reducción espacial: 128->64]
    2. (32, 64, 64) -> Conv2d -> (64, 32, 32)          [Reducción espacial: 64->32]
    3. (64, 32, 32) -> Conv2d -> (128, 16, 16)         [Reducción espacial: 32->16]
    4. (128, 16, 16) -> Conv2d -> (256, 8, 8)          [Reducción espacial: 16->8]
    
    La dimensión del espacio latente es: 256 * 8 * 8 = 16,384
    
    Decoder:
    --------
    Proceso inverso usando ConvTranspose2d para upsampling
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )


    def forward(self, x):
        """
        Forward pass del modelo.
        
        Proceso:
        1. La imagen pasa por el encoder -> representación comprimida
        2. La representación comprimida pasa por el decoder -> reconstrucción
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_epoch(model, dataloader, criterion, optimizer, device):
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
    model.train()
    running_loss = 0.0
    for batch, _ in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
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
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, is_best=False, checkpoint_dir='./checkpoints'):
    """
    Guarda un checkpoint del modelo
    
    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Número de epoch
        loss: Pérdida
        is_best: Si es el mejor modelo
        checkpoint_dir: Directorio para guardar el checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{epoch}.pth"
    print(f"Saving checkpoint at epoch {epoch}")
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = f"{checkpoint_dir}/best_model.pth"
        print(f"Saving best model at epoch {epoch}")
        torch.save(checkpoint, best_path)
        # clean up old checkpoints
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".pth") and file != f"checkpoint_{epoch}.pth" and file != "best_model.pth":
                os.remove(f"{checkpoint_dir}/{file}")

'''
def save_checkpoint(model, optimizer, epoch, loss, is_best=False, checkpoint_dir='./checkpoints'):
    # Guardar solo los pesos del modelo
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = f"{checkpoint_dir}/best_model.pth"
        torch.save(checkpoint, best_path)
'''

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

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, checkpoint_freq=5):
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
    best_val_loss = float('inf')

    if os.path.exists(paths.AUTOENCODER_CHECKPOINT_DIR):
        # check if there is a best_checkpoint and load it if not, load the last checkpoint
        if os.path.exists(f"{paths.AUTOENCODER_CHECKPOINT_DIR}/best_model.pth"):
            model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, f"{paths.AUTOENCODER_CHECKPOINT_DIR}/best_model.pth")
            print(f"Loaded best model from epoch {start_epoch}")
        else:
            model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, f"{paths.AUTOENCODER_CHECKPOINT_DIR}/checkpoint_{epochs-1}.pth")
            print(f"Loaded model from epoch {start_epoch}")
    else:
        os.makedirs(paths.AUTOENCODER_CHECKPOINT_DIR)
        start_epoch = 0


    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,checkpoint_dir=paths.AUTOENCODER_CHECKPOINT_DIR)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, is_best=True, checkpoint_dir=paths.AUTOENCODER_CHECKPOINT_DIR)

    # save final model
    save_checkpoint(model, optimizer, epoch, val_loss, is_best=False, checkpoint_dir=paths.AUTOENCODER_MODEL_DIR) 
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
    paths.setup_autoencoder_paths()
    # Configurar dispositivo
    device = get_device()
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Definir transformaciones
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Obtener dataloaders
    train_loader, val_loader, test_loader = lfw.get_data_loaders(
        batch_size=32,
        base_dir=paths.BASE_DATA_DIR,
        download=False,
        transform=transform
    )

    # Crear modelo
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Entrenar modelo
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=30, checkpoint_freq=5)

if __name__ == '__main__':
    main()