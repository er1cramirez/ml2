import os
import zipfile
import numpy as np
import requests
import torch
from torch.utils.data import Dataset

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(384, 384)):
        """
        Args:
            image_dir (str): Directorio con las imágenes
            mask_dir (str): Directorio con las máscaras
            transform: Transformaciones opcionales
            img_size (tuple): Tamaño objetivo para las imágenes y máscaras.
                            Por defecto 384x384 (divisible por 16 para U-Net)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size

        # Obtener listas de archivos
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

        # Verificar que tenemos el mismo número de imágenes y máscaras
        if len(self.images) != len(self.masks):
            raise ValueError(f"Número diferente de imágenes ({len(self.images)}) y máscaras ({len(self.masks)})")

        # Crear mapeo entre imágenes y máscaras
        self.image_mask_pairs = []
        for img_file in self.images:
            img_num = int(img_file.split('_')[1].split('.')[0])
            mask_file = f'segm_{img_num}.npy'

            if mask_file in self.masks:
                self.image_mask_pairs.append((img_file, mask_file))

        print(f"Dataset cargado con {len(self.image_mask_pairs)} pares de imagen-máscara")

    def resize_array(self, array):
        """Redimensiona un array al tamaño objetivo usando zoom"""
        from scipy.ndimage import zoom

        # Calcular factores de escala
        scale_factors = (self.img_size[0] / array.shape[0],
                        self.img_size[1] / array.shape[1])

        return zoom(array, scale_factors, order=1)  # order=1 para interpolación lineal

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.image_mask_pairs[idx]

        # Cargar archivos
        image = np.load(os.path.join(self.image_dir, img_file))
        mask = np.load(os.path.join(self.mask_dir, mask_file))

        # Redimensionar si es necesario
        if image.shape != self.img_size:
            image = self.resize_array(image)
            mask = self.resize_array(mask)

        # Normalizar
        image = image / image.max()  # Normalización a [0,1]
        mask = (mask > 0).astype(np.float32)  # Binarizar máscara

        # Convertir a tensores
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def setup_dataset(base_dir, download, **dataset_kwargs):
    """
    Configura el dataset MRIs con descarga automática

    Args:
        base_dir (str): Directorio base para almacenar los datos
        download (bool): Si debe descargarse el dataset
        dataset_kwargs (dict): Argumentos específicos del dataset

    Returns:
        str: Ruta al directorio de imágenes
    """
    # Descargar y extraer el dataset si es necesario
    url = 'https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/MRIs.zip'
    data_path = os.path.join(base_dir, "MRIs")
    # verificamos si ya existe el dataset
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("Descargando dataset...")
        response = requests.get(url)
        data_path = base_dir
        zip_path = os.path.join(data_path, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Extrayendo archivos...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_path)
        data_path = os.path.join(base_dir, "MRIs")
        # rename MRIs folder to images and Segmentations to masks
        os.rename(os.path.join(data_path, "MRIs"), os.path.join(data_path, "images"))
        os.rename(os.path.join(data_path, "Segmentations"), os.path.join(data_path, "masks"))
        print("Dataset preparado exitosamente.")
    else:
        print("Dataset encontrado en disco.")
    return data_path


def get_dataloaders(
    batch_size=32,
    base_dir='./data',
    download=True,
    transform=None,
    split_sizes=(0.8, 0.1, 0.1),
    num_workers=None,
    **dataset_kwargs
):
    """
    Crea dataloaders para un dataset de imágenes y máscaras

    Args:
        batch_size (int): Tamaño del batch
        base_dir (str): Directorio base para almacenar los datos
        download (bool): Si debe descargarse el dataset
        transform: Transformaciones opcionales
        split_sizes (tuple): Proporciones para dividir el dataset en entrenamiento, validación y prueba
        num_workers (int): Número de workers para cargar los datos
    """
    # Configurar el dataset
    data_dir = setup_dataset(base_dir, download, **dataset_kwargs)
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    # Transformaciones por defecto si no se especifican
    if transform is None:
        transform = torch.nn.Identity()

    # Configurar número de workers
    if num_workers is None:
        num_workers = 4 if torch.cuda.is_available() else 0

    # Crear dataset
    full_dataset = ImageMaskDataset(image_dir, mask_dir, transform=transform)

    # Dividir dataset
    train_size = int(split_sizes[0] * len(full_dataset))
    val_size = int(split_sizes[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    # Crear dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

    