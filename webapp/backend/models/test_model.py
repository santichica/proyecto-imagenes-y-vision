from pathlib import Path
import torch
import timm
from PIL import Image
from torchvision import transforms


# 🔥 Transform (igual que entrenamiento)
_EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class Classifier:
    def __init__(self, model_path: str | Path, device: str | None = None):
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 🔥 Arquitectura
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=2
        )

        # 🔥 Cargar pesos
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Modelo cargado en {self.device}")


    def predict(self, image_path: str | Path) -> dict:
        img = Image.open(image_path).convert('RGB')
        x = _EVAL_TF(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        return {
            'label': 'melanoma' if probs[1] > probs[0] else 'nevus',
            'prob_mel': round(float(probs[1]), 4),
            'prob_nv': round(float(probs[0]), 4),
        }


# 🔥 TEST LOCAL
if __name__ == "__main__":
    # 👉 Ajusta rutas si es necesario
    model_path = "best_model_lora.pt"
    image_path = "test_image.png"

    clf = Classifier(model_path)

    result = clf.predict(image_path)

    print("\n🧪 Resultado:")
    print(result)