# Generación sintética de melanoma

Flujo en dos notebooks: entrenamiento de Textual Inversion y generación de imágenes.
Compatible con Google Colab (CUDA), Mac Apple Silicon (MPS) y CPU.

---

## Requisitos

```bash
conda activate ham10000-augmentation
```

Los notebooks instalan `diffusers`, `transformers`, `accelerate`, `safetensors` y
`huggingface_hub` automáticamente si no están presentes.

---

## Paso 0 — Preparar los datos

Desde la raíz del proyecto:

```bash
python scripts/augmentation/prepare_for_colab.py
```

Crea `data/processed/melanoma_train_for_colab.zip` (~232 MB) con las 801 imágenes
de melanoma del split de entrenamiento.

**Si usas Google Colab**: sube el ZIP a Google Drive en:
`Mi unidad/ham10000-augmentation/melanoma_train_for_colab.zip`

---

## Paso 1 — Textual Inversion

Abre **`HAM10000_textual_inversion.ipynb`** y ejecuta todas las celdas en orden.

Aprende el token `<mel-skin>` en 5 000 pasos de fine-tuning del text encoder de
Stable Diffusion v1.5.

**Tiempos estimados**

| Hardware | Tiempo |
|---|---|
| Google Colab T4 (16 GB VRAM) | ~30 min |
| Apple Silicon M1/M2 (MPS) | ~2–4 h |
| CPU | ~6–12 h |

Para una prueba rápida cambia `TI_STEPS = 200` en la celda de hiperparámetros.

**Salidas locales**
```
data/processed/ti_models/mel_skin_embedding_final.pt
data/processed/ti_models/mel_skin_embedding_step500.pt  (checkpoints cada 500 steps)
```

**Salidas en Colab (Google Drive)**
```
Mi unidad/ham10000-augmentation/models/mel_skin_embedding_final.pt
```

---

## Paso 2 — Generación

Abre **`HAM10000_generation.ipynb`** y ejecuta todas las celdas.

Genera dos lotes de imágenes sintéticas:

| Método | Imágenes | Tiempo (T4) |
|---|---|---|
| Textual Inversion (nuevas desde ruido) | 4 500 | ~90 min |
| Img2Img (variaciones de reales) | 2 403 | ~30 min |

**Salidas**
```
data/synthetic/textual_inversion/ti_NNNNN_sSSSSS.jpg
data/synthetic/img2img/i2i_ISIC_XXXXXXXX_strN_sSSSSS.jpg
data/synthetic/generation_metadata.json
```

**Si usas Colab**: descarga la carpeta `synthetic/` de Google Drive al directorio
`data/` del proyecto local después de que termine.

---

## Notas por hardware

### Apple Silicon (MPS)
- GPU detectada con `torch.backends.mps.is_available()`
- Se usa `float32` en lugar de `float16` (evita NaN en algunos ops de MPS)
- El generator de difusión se crea en CPU (requerimiento de diffusers con MPS)
- `BATCH_SIZE=1` por defecto; puedes subir a 2 con ≥16 GB de RAM unificada
- Lanza Jupyter desde la raíz del proyecto para que los paths relativos funcionen

### Google Colab (CUDA T4)
- `float16` y `batch_size=4` — generación más rápida
- Los checkpoints de TI se guardan en Drive cada 500 steps (recuperable si expira la sesión)

### CPU
- Funcional pero muy lento; solo útil para pruebas con `TI_STEPS=200`
