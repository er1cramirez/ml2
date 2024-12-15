import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.datasets as datasets
import urllib.request
import tarfile

def setup_lfw_dataset(base_dir='./data', download=True, **dataset_kwargs):
    """
    Configura cualquier dataset con descarga automática.
    
    Args:
        base_dir (str): Directorio base para almacenar los datos
        download (bool): Si debe descargarse el dataset
        dataset_kwargs (dict): Argumentos específicos del dataset
    
    Returns:
        str: Ruta al directorio de datos
    """
    dataset_name = 'lfw'
    # Mapeo de datasets a sus URLs y funciones de procesamiento
    DATASET_CONFIGS = {
        'lfw': {
            'url': "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
            'processor': _process_tgz
        },
        # Añadir más datasets aquí
    }
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} no soportado")
        
    config = DATASET_CONFIGS[dataset_name]
    data_dir = os.path.join(base_dir, dataset_name)
    
    # Crear directorio base si no existe
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Verificar si el dataset existe
    if not os.path.exists(data_dir) and download:
        print(f"Descargando dataset {dataset_name}...")
        file_path = os.path.join(base_dir, f'{dataset_name}.tgz')
        urllib.request.urlretrieve(config['url'], file_path)
        
        print("Extrayendo archivos...")
        config['processor'](file_path, base_dir)
        os.remove(file_path)
        print(f"Dataset {dataset_name} preparado correctamente.")
    elif not os.path.exists(data_dir) and not download:
        raise RuntimeError("Dataset no encontrado. Establece download=True para descargarlo.")
    else:
        print(f"Dataset {dataset_name} encontrado en disco.")
    
    return data_dir

def get_data_loaders(
    batch_size=32,
    base_dir='./data',
    download=True,
    transform=None,
    split_sizes=(0.8, 0.1, 0.1),
    num_workers=None,
    **dataset_kwargs
):
    """
    Crea dataloader
    
    Args:
        batch_size (int): Tamaño del batch
        base_dir (str): Directorio base para los datos
        download (bool): Si debe descargarse el dataset
        transform (callable): Transformaciones a aplicar
        split_sizes (tuple): Proporciones para train/val/test
        num_workers (int): Número de workers para DataLoader
        dataset_kwargs (dict): Argumentos específicos del dataset
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Configurar el dataset
    data_dir = setup_lfw_dataset(base_dir, download, **dataset_kwargs)
    
    # Transformaciones por defecto si no se especifican
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    # Configurar número de workers
    if num_workers is None:
        num_workers = 4 if torch.cuda.is_available() else 0
        
    # Crear dataset
    dataset = ImageFolder(root=data_dir, transform=transform)
    
    # Validar y aplicar split_sizes
    if sum(split_sizes) != 1.0:
        raise ValueError("Las proporciones del split deben sumar 1.0")
    
    # Calcular tamaños
    total_size = len(dataset)
    train_size = int(split_sizes[0] * total_size)
    val_size = int(split_sizes[1] * total_size)
    test_size = total_size - train_size - val_size
    
    # Dividir dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Configuración común para los dataloaders
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available()
    }
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader

def _process_tgz(file_path, extract_dir):
    """Función auxiliar para procesar archivos .tgz"""
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)