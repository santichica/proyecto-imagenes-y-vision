import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import yaml
from pathlib import Path
from datetime import datetime 
# Priorizamos torch-fidelity como fue requerido y por ser la mejor opción funcional
try:
    from torch_fidelity import calculate_metrics
except ImportError:
    raise ImportError("Para ejecutar este script necesitas instalar torch-fidelity.\n👉 Ejecuta: pip install torch-fidelity")

class SafeImageDataset(Dataset):
    """
    Dataset robusto para cargar imágenes. Detecta y filtra silenciosamente
    imágenes corruptas o no soportadas para no romper la ejecución de la métrica.
    Devuelve los tensores en formato uint8 (requerido por torch-fidelity).
    """
    def __init__(self, folder_path, ext_filter=('.png', '.jpg', '.jpeg', '.webp'), size=(299, 299)):
        self.folder_path = Path(folder_path)
        self.valid_paths = []
        self.size = size
        
        if not self.folder_path.exists():
            print(f"Error: La ruta {folder_path} no existe.")
            return
            
        for file in self.folder_path.iterdir():
            if file.is_file() and file.suffix.lower() in ext_filter:
                try:
                    # Test rápido para validar integridad del archivo
                    with Image.open(file) as img:
                        img.verify()
                    self.valid_paths.append(file)
                except (Exception, UnidentifiedImageError):
                    print(f"[Advertencia] Ignorando archivo inválido: {file.name}")
                    
    def __len__(self):
        return len(self.valid_paths)
        
    def __getitem__(self, idx):
        path = self.valid_paths[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            # Resizing to ensure all images have the same dimensions for collation
            img = img.resize(self.size, Image.BILINEAR)
            # Convierte directamente a torch.uint8 de forma eficiente
            tensor = F.pil_to_tensor(img)
        return tensor

def evaluate_generation(real_dir: str, gen_dir: str, batch_size: int = 32):
    """
    Evalúa las imágenes generadas vs reales obteniendo FID y el Inception Score (IS).
    """
    real_dataset = SafeImageDataset(real_dir)
    gen_dataset = SafeImageDataset(gen_dir)
    
    n_real = len(real_dataset)
    n_gen = len(gen_dataset)
    
    if n_real == 0 or n_gen == 0:
        print("Error: No hay suficientes imágenes válidas en los directorios especificados.")
        return
        
    # Verificar disponibilidad de GPU (Evita cuellos de botella e incompatibility)
    device_is_cuda = torch.cuda.is_available()
    
    # Calcular métricas usando torch-fidelity
    # - isc: Inception Score
    # - fid: Fréchet Inception Distance
    metrics = calculate_metrics(
        input1=gen_dataset,     # Generadas (para IS y comp. FID)
        input2=real_dataset,    # Reales (como base para FID)
        cuda=device_is_cuda,
        isc=True,
        fid=True,
        batch_size=batch_size,
        verbose=False # Silenciamos el log recargado para limpieza
    )
    
    # Formateo de salida estandarizado
    print("\n" + "=" * 33)
    print("Image Generation Evaluation")
    print("=" * 33)
    print(f"Real images: {n_real}")
    print(f"Generated images: {n_gen}\n")
    print(f"FID Score: {metrics['frechet_inception_distance']:.2f}")
    print(f"Inception Score: {metrics['inception_score_mean']:.2f}")
    print("=" * 33 + "\n")
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    archivo  = Path(f"evaluation_results_{fecha_hora}.txt")
    if not archivo.exists():
        archivo.touch()

    with open(archivo, "w") as f:
        f.write(f"Evaluation conducted on: {fecha_hora}\n\n")
        f.write("Image Generation Evaluation\n")
        f.write("=" * 33 + "\n")
        f.write(f"Real images: {n_real}\n")
        f.write(f"Generated images: {n_gen}\n\n")
        f.write(f"FID Score: {metrics['frechet_inception_distance']:.2f}\n")
        f.write(f"Inception Score: {metrics['inception_score_mean']:.2f}\n")
        f.write("=" * 33 + "\n")
    
    return metrics

def get_config():
    """Carga configuración del YAML central como se acostumbra en el proyecto."""
    config_path = Path("config/project.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

if __name__ == '__main__':
    # Configuración por defecto siguiendo el flujo del YAML
    config = get_config()
    data_processed = config.get("paths", {}).get("data_processed", "data/processed")
    
    # Rutas por defecto en base a la arquitectura actual (se asumen como ejemplo)
    REAL_IMG_DIR = os.path.join(data_processed, "HAM10000_images_part_1")
    GEN_IMG_DIR = os.path.join(data_processed, "gan_generated") # O la ruta de las sintetizadas
    
    BATCH_SIZE = config.get("modeling", {}).get("batch_size", 32)
    
    if Path(REAL_IMG_DIR).exists() and Path(GEN_IMG_DIR).exists():
        print("Iniciando evaluación de generación de imágenes...")
        evaluate_generation(REAL_IMG_DIR, GEN_IMG_DIR, batch_size=BATCH_SIZE)
    else:
        print("Rutas configuradas no encontradas. Verifica tus directorios:")
        print(f"- Real: {REAL_IMG_DIR}")
        print(f"- Generado: {GEN_IMG_DIR}")
