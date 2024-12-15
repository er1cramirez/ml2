import torch
import torch.nn as nn


"""
Módulo Discriminador
-------------------
Arquitectura convolucional que evalúa si una imagen es real o generada:
- Entrada: Imágenes RGB 64x64
- Capas intermedias: Convoluciones con stride=2 que reducen dimensionalidad
- BatchNorm y LeakyReLU para estabilidad y no-linealidad
- Salida: Probabilidad de que la imagen sea real (0-1)
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            """Bloque básico del discriminador"""
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

"""
Módulo Generador
---------------
Arquitectura que transforma ruido aleatorio en imágenes:
- Entrada: Vector de ruido latent_dim-dimensional
- Capas intermedias: Convoluciones transpuestas que aumentan dimensionalidad
- BatchNorm y ReLU para estabilidad y no-linealidad
- Salida: Imágenes RGB 64x64 con valores en [-1,1]
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 100
        def generator_block(in_channels, out_channels, normalize=True):
            """Bloque básico del generador"""
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            # Primera capa - proyección y reshape del vector latente
            nn.ConvTranspose2d(self.latent_dim, 512, 4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Capas de upsampling
            *generator_block(512, 256),
            *generator_block(256, 128),
            *generator_block(128, 64),
            # Capa final
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.model(z)
