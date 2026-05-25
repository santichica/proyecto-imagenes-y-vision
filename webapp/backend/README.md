# Webapp — Backend

Dos componentes independientes con entornos separados:

| Componente | Archivo | Framework | Modelo |
|---|---|---|---|
| Generación GAN | `main.py` | TensorFlow 2.15 | `models/generator_final.h5` |
| Clasificación | `classifier.py` | PyTorch + timm | `models/best_model.pt` |

---

## 1. Servidor de generación GAN

### Entorno

```bash
conda create -n gan-api python=3.10
conda activate gan-api
pip install -r requirements.txt
```

### Levantar el servidor

```bash
cd webapp/backend
conda activate gan-api
python -m uvicorn main:app --reload
```

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `GET` | `/generate?num_images=5` | Genera N imágenes con el WGAN-GP |

Las imágenes se guardan en `webapp/backend/generated_by_user/` y se sirven como archivos estáticos en `/generated_by_user/<filename>`.

### Modelo serializado

`models/generator_final.h5` — generador WGAN-GP entrenado 100 épocas sobre melanoma HAM10000, imágenes 64×64 px. Se carga con:

```python
import tensorflow as tf
generator = tf.keras.models.load_model('models/generator_final.h5', compile=False)
noise = tf.random.normal([N, 100])          # latent_dim = 100
images = generator(noise, training=False)   # salida: (N, 64, 64, 3), rango [-1, 1]
```

---

## 2. Módulo de clasificación

`classifier.py` expone la clase `Classifier` para integrar el clasificador EfficientNet-B0 en cualquier endpoint FastAPI.

### Entorno (separado del GAN)

```bash
conda create -n clf-api python=3.10
conda activate clf-api
pip install torch torchvision timm Pillow
```

### Uso

```python
from classifier import Classifier

clf = Classifier('models/best_model.pt')
result = clf.predict('imagen.jpg')
# {'label': 'melanoma', 'prob_mel': 0.8312, 'prob_nv': 0.1688}
```

### Modelo serializado

`models/best_model.pt` — EfficientNet-B0 fine-tuned sobre HAM10000 (clasificación binaria nevus/melanoma). Se carga con:

```python
import torch, timm

model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
model.eval()
```

El transform de inferencia debe ser exactamente:

```python
from torchvision import transforms

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```
