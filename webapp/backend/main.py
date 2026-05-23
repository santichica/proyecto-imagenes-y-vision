import os
import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "front"), html=True), name="static")

model_path = os.path.join(os.path.dirname(__file__), "models", "generator_final.h5")

generator = tf.keras.models.load_model(model_path, compile=False)

print(f"✅ Cargando modelo desde: {model_path}")




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

@app.get("/health")
def health_check():
    return {"status": "GAN ready to generate"}

@app.get("/generate")
def generate(num_images: int = 5):
    paths = generate_images(num_images)
    return {"images": paths}