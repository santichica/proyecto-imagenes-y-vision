import os
import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import torch
import timm
from PIL import Image
from torchvision import transforms
from fastapi import UploadFile, File
import io

os.makedirs("generated_by_user", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount("/generated_by_user", StaticFiles(directory="generated_by_user"), name="generated_by_user")

LATENT_DIM = 100

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.mount("/app", StaticFiles(directory=os.path.join(BASE_DIR, "front"), html=True), name="static")

model_path = os.path.join(os.path.dirname(__file__), "models", "generator_final.h5")

generator = tf.keras.models.load_model(model_path, compile=False)

print(f"✅ Modelo GAN cargado desde: {model_path}")

_EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

classifier_model = timm.create_model(
    'efficientnet_b0',
    pretrained=False,
    num_classes=2
)

classifier_model.load_state_dict(
    torch.load("models/best_model_lora.pt", map_location="cpu")
)

classifier_model.eval()




def generate_images(num_images):
    paths = []

    noise = tf.random.normal([num_images, LATENT_DIM])
    predictions = generator(noise, training=False)

    for i in range(num_images):
        # Normalizar de [-1,1] → [0,255]
        img = (predictions[i] + 1) / 2
        img = tf.clip_by_value(img, 0, 1)
        img = tf.image.convert_image_dtype(img, tf.uint8)

        filename = f"generated_by_user/{int(time.time())}_{i}.png"

        # 💾 Guardar imagen
        tf.io.write_file(
            filename,
            tf.image.encode_png(img)
        )

        print(f"✅ Imagen guardada: {filename}")
        paths.append(filename)

    return paths

@app.get("/api/health")
def health_check():
    return {"status": "GAN ready to generate"}

@app.get("/api/generate")
def generate(num_images: int = 5):
    paths = generate_images(num_images)
    return {"images": paths}

@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    x = _EVAL_TF(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(classifier_model(x), dim=1)[0]

    return {
        "label": "melanoma" if probs[1] > probs[0] else "nevus",
        "prob_mel": round(float(probs[1]), 4),
        "prob_nv": round(float(probs[0]), 4),
    }