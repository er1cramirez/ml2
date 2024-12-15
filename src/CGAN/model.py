import torch
import torch.nn as nn

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

