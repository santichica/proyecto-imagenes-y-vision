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
| Real + TI (2×) | 0.920 | 0.585 | **0.616** |
| Real + LoRA (2×) | 0.926 | 0.535 | 0.607 |
| Real + WGAN-GP (2×) | **0.929** | 0.579 | 0.605 |
| Real + Derm-T2IM s=0.40 (2×) | 0.917 | 0.604 | 0.585 |
| Real + Derm-T2IM s=0.05 (2×) | 0.920 | **0.610** | 0.614 |
| Synthetic only — TI | 0.561 | 0.000 | 0.000 |
| Synthetic only — LoRA | 0.551 | 0.006 | 0.012 |
| Synthetic only — WGAN-GP | 0.610 | 0.006 | 0.012 |
| Synthetic only — Derm-T2IM s=0.40 | 0.845 | 0.145 | 0.236 |
| Synthetic only — Derm-T2IM s=0.05 | 0.878 | 0.478 | 0.522 |

**Conclusiones:** la augmentación híbrida mejora el AUC en todos los generadores (+0.6–1.8 pp); Derm-T2IM s=0.05 registra el mayor Recall (+8.2 pp). En entrenamiento exclusivamente sintético, solo Derm-T2IM mantiene viabilidad: s=0.05 (FID=21.1) obtiene AUC 0.878 y Recall 0.478; s=0.40 (FID=46.7) evita el colapso total pero con Recall bajo (0.145). TI, LoRA y WGAN-GP colapsan (Recall≈0). El FID es el indicador más predictivo de la transferibilidad del clasificador.

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

## Estructura del repositorio

```
proyecto-imagenes-y-vision/
├── HAM10000_classification_comparative.ipynb  ← notebook principal (11 escenarios)
├── HAM10000_quality_evaluation.ipynb          ← FID e IS por generador
├── HAM10000_paper_figures.ipynb               ← figuras para el artículo
├── GAN/
│   ├── HAM10000_GAN.ipynb                     ← entrenamiento WGAN-GP
│   └── checkpoints/generator_final.h5         ← pesos del generador
├── notebooks/
│   ├── generation/
│   │   ├── HAM10000_textual_inversion.ipynb   ← entrenamiento token TI
│   │   ├── HAM10000_generation.ipynb          ← generación SD (TI + img2img)
│   │   ├── HAM10000_lora_training.ipynb       ← fine-tuning LoRA
│   │   └── HAM10000_derm_generation.ipynb     ← generación Derm-T2IM
│   ├── experiments/                           ← versiones previas del clasificador
│   └── exploration/                           ← EDA y baseline exploratorio
├── scripts/
│   ├── data_processing/
│   │   ├── 01_extract.py                      ← extracción de imágenes HAM10000
│   │   └── 02_split.py                        ← split lesion-aware 70/15/15
│   ├── augmentation/
│   │   └── evaluate_generation.py             ← cálculo de FID e IS
│   └── training/                              ← dataset.py, model.py, train.py
├── data/
│   ├── processed/                             ← splits CSV + classification_data.zip
│   └── synthetic/                             ← imágenes sintéticas locales
├── reports/                                   ← figuras y métricas de calidad
├── config/project.yaml                        ← rutas y parámetros globales
├── webapp/
│   ├── backend/                               ← FastAPI (GAN + clasificador)
│   └── front/                                 ← interfaz HTML/JS
├── tests/test_split_leakage.py
├── environment.yml
└── requirements.txt
```

Los datos completos (imágenes sintéticas, modelos entrenados, runs de experimentos) están en [Google Drive](https://drive.google.com/drive/folders/1dAEHEQwjZaMYovx--IB0l727m25F_FRV?usp=drive_link).

---

> **Nota:** este proyecto fue desarrollado con asistencia de [Claude Code](https://claude.ai/code) (Anthropic) para tareas de implementación, depuración y documentación.
