# Configuracion y/o creacion de rutas para guardar checkpoints, modelos, logs, etc.

import os

# Directorio base datasets
BASE_DATA_DIR = '/home/eric/ml2/data'

# LFW
LFW_DIR = os.path.join(BASE_DATA_DIR, 'lfw')

# MNIST
MNIST_DIR = os.path.join(BASE_DATA_DIR, 'mnist')

# CIFAR10
CIFAR10_DIR = os.path.join(BASE_DATA_DIR, 'cifar10')

# CelebA
CELEBA_DIR = os.path.join(BASE_DATA_DIR, 'celeba')

# Imagenet
IMAGENET_DIR = os.path.join(BASE_DATA_DIR, 'imagenet')

# MRIs
MRIS_DIR = os.path.join(BASE_DATA_DIR, 'MRIs')

# Directorio base checkpoints
BASE_CHECKPOINT_DIR = 'out/checkpoints'

# Directorio base modelos
BASE_MODEL_DIR = 'out/models'

# Directorio base backups
BASE_BACKUP_DIR = 'backups'

# Autoencoder
AUTOENCODER_DIR = '/home/eric/ml2/src/Autoencoder'
AUTOENCODER_CHECKPOINT_DIR = os.path.join(AUTOENCODER_DIR, BASE_CHECKPOINT_DIR)
AUTOENCODER_MODEL_DIR = os.path.join(AUTOENCODER_DIR, BASE_MODEL_DIR)
AUTOENCODER_BACKUP_DIR = os.path.join(AUTOENCODER_DIR, BASE_BACKUP_DIR)
def setup_autoencoder_paths():
    if not os.path.exists(AUTOENCODER_CHECKPOINT_DIR):
        os.makedirs(AUTOENCODER_CHECKPOINT_DIR)
    if not os.path.exists(AUTOENCODER_MODEL_DIR):
        os.makedirs(AUTOENCODER_MODEL_DIR)
    if not os.path.exists(AUTOENCODER_BACKUP_DIR):
        os.makedirs(AUTOENCODER_BACKUP_DIR)

# GAN
GAN_DIR = '/home/eric/ml2/src/GAN'
GAN_CHECKPOINT_DIR = os.path.join(GAN_DIR, BASE_CHECKPOINT_DIR)
GAN_MODEL_DIR = os.path.join(GAN_DIR, BASE_MODEL_DIR)
GAN_BACKUP_DIR = os.path.join(GAN_DIR, BASE_BACKUP_DIR)
def setup_gan_paths():
    if not os.path.exists(GAN_CHECKPOINT_DIR):
        os.makedirs(GAN_CHECKPOINT_DIR)
    if not os.path.exists(GAN_MODEL_DIR):
        os.makedirs(GAN_MODEL_DIR)
    if not os.path.exists(GAN_BACKUP_DIR):
        os.makedirs(GAN_BACKUP_DIR)

# GAN_FID
GAN_FID_DIR = '/home/eric/ml2/src/GAN_FID'
GAN_FID_CHECKPOINT_DIR = os.path.join(GAN_FID_DIR, BASE_CHECKPOINT_DIR)
GAN_FID_MODEL_DIR = os.path.join(GAN_FID_DIR, BASE_MODEL_DIR)
GAN_FID_BACKUP_DIR = os.path.join(GAN_FID_DIR, BASE_BACKUP_DIR)
def setup_gan_fid_paths():
    if not os.path.exists(GAN_FID_CHECKPOINT_DIR):
        os.makedirs(GAN_FID_CHECKPOINT_DIR)
    if not os.path.exists(GAN_FID_MODEL_DIR):
        os.makedirs(GAN_FID_MODEL_DIR)
    if not os.path.exists(GAN_FID_BACKUP_DIR):
        os.makedirs(GAN_FID_BACKUP_DIR)

# CGAN
CGAN_DIR = './CGAN'
CGAN_CHECKPOINT_DIR = os.path.join(CGAN_DIR, BASE_CHECKPOINT_DIR)
CGAN_MODEL_DIR = os.path.join(CGAN_DIR, BASE_MODEL_DIR)
CGAN_BACKUP_DIR = os.path.join(CGAN_DIR, BASE_BACKUP_DIR)
def setup_cgan_paths():
    if not os.path.exists(CGAN_CHECKPOINT_DIR):
        os.makedirs(CGAN_CHECKPOINT_DIR)
    if not os.path.exists(CGAN_MODEL_DIR):
        os.makedirs(CGAN_MODEL_DIR)
    if not os.path.exists(CGAN_BACKUP_DIR):
        os.makedirs(CGAN_BACKUP_DIR)

# BEGAN
BEGAN_DIR = '/home/eric/ml2/src/BEGAN'
BEGAN_CHECKPOINT_DIR = os.path.join(BEGAN_DIR, BASE_CHECKPOINT_DIR)
BEGAN_MODEL_DIR = os.path.join(BEGAN_DIR, BASE_MODEL_DIR)
BEGAN_BACKUP_DIR = os.path.join(BEGAN_DIR, BASE_BACKUP_DIR)
def setup_began_paths():
    if not os.path.exists(BEGAN_CHECKPOINT_DIR):
        os.makedirs(BEGAN_CHECKPOINT_DIR)
    if not os.path.exists(BEGAN_MODEL_DIR):
        os.makedirs(BEGAN_MODEL_DIR)
    if not os.path.exists(BEGAN_BACKUP_DIR):
        os.makedirs(BEGAN_BACKUP_DIR)

# UNET
UNET_DIR = './UNET'
UNET_CHECKPOINT_DIR = os.path.join(UNET_DIR, BASE_CHECKPOINT_DIR)
UNET_MODEL_DIR = os.path.join(UNET_DIR, BASE_MODEL_DIR)
UNET_BACKUP_DIR = os.path.join(UNET_DIR, BASE_BACKUP_DIR)
def setup_unet_paths():
    if not os.path.exists(UNET_CHECKPOINT_DIR):
        os.makedirs(UNET_CHECKPOINT_DIR)
    if not os.path.exists(UNET_MODEL_DIR):
        os.makedirs(UNET_MODEL_DIR)
    if not os.path.exists(UNET_BACKUP_DIR):
        os.makedirs(UNET_BACKUP_DIR)

# Pix2Pix
PIX2PIX_DIR = './Pix2Pix'
PIX2PIX_CHECKPOINT_DIR = os.path.join(PIX2PIX_DIR, BASE_CHECKPOINT_DIR)
PIX2PIX_MODEL_DIR = os.path.join(PIX2PIX_DIR, BASE_MODEL_DIR)
PIX2PIX_BACKUP_DIR = os.path.join(PIX2PIX_DIR, BASE_BACKUP_DIR)
def setup_pix2pix_paths():
    if not os.path.exists(PIX2PIX_CHECKPOINT_DIR):
        os.makedirs(PIX2PIX_CHECKPOINT_DIR)
    if not os.path.exists(PIX2PIX_MODEL_DIR):
        os.makedirs(PIX2PIX_MODEL_DIR)
    if not os.path.exists(PIX2PIX_BACKUP_DIR):
        os.makedirs(PIX2PIX_BACKUP_DIR)

# CycleGAN
CYCLEGAN_DIR = './CycleGAN'
CYCLEGAN_CHECKPOINT_DIR = os.path.join(CYCLEGAN_DIR, BASE_CHECKPOINT_DIR)
CYCLEGAN_MODEL_DIR = os.path.join(CYCLEGAN_DIR, BASE_MODEL_DIR)
CYCLEGAN_BACKUP_DIR = os.path.join(CYCLEGAN_DIR, BASE_BACKUP_DIR)
def setup_cyclegan_paths():
    if not os.path.exists(CYCLEGAN_CHECKPOINT_DIR):
        os.makedirs(CYCLEGAN_CHECKPOINT_DIR)
    if not os.path.exists(CYCLEGAN_MODEL_DIR):
        os.makedirs(CYCLEGAN_MODEL_DIR)
    if not os.path.exists(CYCLEGAN_BACKUP_DIR):
        os.makedirs(CYCLEGAN_BACKUP_DIR)


