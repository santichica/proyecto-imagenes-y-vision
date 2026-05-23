"""
Inferencia de clasificación de lesiones cutáneas con EfficientNet-B0.

Uso:
    from classifier import Classifier
    clf = Classifier('models/best_model.pt')
    result = clf.predict('imagen.jpg')
    # {'label': 'melanoma', 'prob_mel': 0.8312, 'prob_nv': 0.1688}

El transform es idéntico a EVAL_TF del notebook de entrenamiento:
    Resize(224, 224) → ToTensor → Normalize(ImageNet mean/std)
"""

from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms


# Transform idéntico a EVAL_TF en HAM10000_classification_comparative.ipynb
_EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class Classifier:
    def __init__(self, model_path: str | Path, device: str | None = None):
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=2
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str | Path) -> dict:
        img = Image.open(image_path).convert('RGB')
        x   = _EVAL_TF(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        return {
            'label':    'melanoma' if probs[1] > probs[0] else 'nevus',
            'prob_mel': round(float(probs[1]), 4),
            'prob_nv':  round(float(probs[0]), 4),
        }
