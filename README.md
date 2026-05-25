# Augmentación sintética para mitigar el desbalanceo de clases en la detección de melanoma dermoscópico

Clasificación binaria **Nevus vs Melanoma** sobre HAM10000 con augmentación sintética via Stable Diffusion y WGAN-GP. Proyecto de investigación que compara cuatro métodos generativos (Textual Inversion, LoRA, Derm-T2IM y WGAN-GP) como estrategia de rebalanceo de clases sin técnicas de sampling artificial.

**Integrantes:** Santiago Chica · Gabriel Gómez · Michael Patiño · Juan Diego Sarmiento

---

## Demo y recursos

| Recurso | Enlace |
|---|---|
| Webapp (generación GAN) | [http://34.205.247.76:8000/app/](http://34.205.247.76:8000/app/) |
| Documentación de la webapp | [webapp/backend/README.md](webapp/backend/README.md) |
| Video de explicación | [YouTube](https://www.youtube.com/watch?v=SQ6f0hyAaiY) |
| Datos, modelos y artefactos (Drive) | [Google Drive](https://drive.google.com/drive/folders/1dAEHEQwjZaMYovx--IB0l727m25F_FRV?usp=drive_link) |

---

## Resultados principales

Clasificador EfficientNet-B0, test set 100% real (159 mel / 1 012 nv):

| Escenario | AUC | Recall mel | F1 mel |
|---|---|---|---|
| Real only (baseline) | 0.911 | 0.528 | 0.579 |
| Real + TI (2×) | 0.920 | 0.585 | 0.616 |
| Real + LoRA (2×) | 0.926 | 0.535 | 0.607 |
| Real + WGAN-GP (2×) | **0.929** | 0.579 | 0.605 |
| Real + Derm-T2IM s=0.40 (2×) | 0.917 | **0.604** | 0.585 |
| Real + Derm-T2IM s=0.05 (2×) | — | — | — |
| Synthetic only — TI | 0.561 | 0.000 | 0.000 |
| Synthetic only — LoRA | 0.551 | 0.006 | 0.012 |
| Synthetic only — WGAN-GP | 0.610 | 0.006 | 0.012 |
| Synthetic only — Derm-T2IM s=0.05 | 0.878 | 0.478 | 0.522 |
| Synthetic only — Derm-T2IM s=0.40 | — | — | — |

**Conclusiones:** la augmentación híbrida mejora el AUC en todos los generadores (+0.6–1.8 pp). En entrenamiento exclusivamente sintético, solo Derm-T2IM con perturbación mínima (strength=0.05, FID=21) mantiene viabilidad; TI, LoRA y WGAN-GP colapsan (Recall≈0) por domain shift severo. El FID es el indicador más predictivo de la transferibilidad del clasificador.

---

## Requisitos

- Anaconda o Miniconda
- GPU NVIDIA con CUDA 13.0 (T4, V100, A100, RTX 30xx+)
- ~15 GB de espacio en disco

---

## Instalación

```bash
conda create --name ham10000-augmentation python=3.11 -y
conda activate ham10000-augmentation

pip install numpy pandas scikit-learn pyyaml pillow matplotlib seaborn tqdm \
            jupyterlab ipykernel pytest

pip install "torch==2.11.0+cu130" torchvision \
            --index-url https://download.pytorch.org/whl/cu130

pip install timm diffusers transformers accelerate safetensors \
            "torchmetrics[image]" python-dotenv

python -m ipykernel install --user \
       --name ham10000-augmentation \
       --display-name "HAM10000 Augmentation"
```

---

## Preparar los datos

```bash
python scripts/data_processing/01_extract.py   # imágenes → data/processed/images/
python scripts/data_processing/02_split.py     # split lesion-aware 70/15/15
pytest tests/test_split_leakage.py -v          # verificar ausencia de data leakage
```

El dataset completo (imágenes reales + sintéticas + modelos entrenados) está disponible en [Google Drive](https://drive.google.com/drive/folders/1dAEHEQwjZaMYovx--IB0l727m25F_FRV?usp=drive_link).

---

## Notebooks principales

| Notebook | Descripción |
|---|---|
| [`HAM10000_classification_comparative.ipynb`](HAM10000_classification_comparative.ipynb) | 11 escenarios de clasificación (baseline · 2× · synthetic-only) |
| [`HAM10000_quality_evaluation.ipynb`](HAM10000_quality_evaluation.ipynb) | FID e IS por generador |
| [`HAM10000_textual_inversion.ipynb`](notebooks/generation/HAM10000_textual_inversion.ipynb) | Entrenamiento del token TI |
| [`HAM10000_generation.ipynb`](notebooks/generation/HAM10000_generation.ipynb) | Generación con SD (TI + img2img) |
| [`HAM10000_lora_training.ipynb`](notebooks/generation/HAM10000_lora_training.ipynb) | Fine-tuning LoRA (rank=32) sobre SD v1.5 |
| [`HAM10000_derm_generation.ipynb`](notebooks/generation/HAM10000_derm_generation.ipynb) | Generación Derm-T2IM (img2img dermoscopy-specific) |
| [`GAN/HAM10000_GAN.ipynb`](GAN/HAM10000_GAN.ipynb) | Entrenamiento WGAN-GP |

Ver diseño metodológico completo en [METHODOLOGY.md](METHODOLOGY.md).

---

## Estructura de Drive (Colab)

```
ham10000-augmentation/
├── data/
│   └── classification_data.zip
├── synthetic/
│   ├── textual_inversion/   ← ~4 500 imágenes (*.jpg)
│   ├── lora/                ← ~4 500 imágenes (*.jpg)
│   ├── gan_final/           ← 5 000 imágenes (*.png)
│   ├── derm_s040/           ← ~2 400 imágenes (*.jpg)
│   └── derm_s005/           ← ~1 600 imágenes (*.jpg)
├── models/
│   └── mel_skin_embedding_final.pt
└── experiments/
    └── YYYYMMDD_HHMMSS_<escenario>/
```

---

> **Nota:** este proyecto fue desarrollado con asistencia de [Claude Code](https://claude.ai/code) (Anthropic) para tareas de implementación, depuración y documentación.
