# HAM10000 Synthetic Augmentation Research

Clasificación binaria **Nevus vs Melanoma** usando HAM10000 con augmentación sintética via Stable Diffusion.

## Requisitos

- [Anaconda](https://www.anaconda.com/download) o Miniconda
- GPU NVIDIA con compute capability ≥ 7.5 (T4, V100, A100, RTX 30xx+) y CUDA 13.0
- ~15 GB de espacio en disco (dataset + imágenes extraídas)

---

## 1. Crear el entorno

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

## 2. Descargar el dataset

El script descarga HAM10000 desde Harvard Dataverse (DOI: `10.7910/DVN/DBW86T`).

```bash
python scripts/download_ham10000.py
```

Los archivos quedan en `data/raw/ham10000/`. La descarga es de ~3.5 GB e incluye las imágenes y los metadatos.

---

## 3. Preparar los datos

```bash
# Extraer imágenes de los ZIPs → data/processed/images/
python scripts/data/01_extract.py

# Split lesion-aware (70/15/15) → data/processed/splits/
python scripts/data/02_split.py

# Verificar que no hay data leakage
pytest tests/test_split_leakage.py -v
```

---

## 4. Correr el baseline

Abre JupyterLab y selecciona el kernel **HAM10000 Augmentation**:

```bash
jupyter lab
```

Abre [HAM10000_baseline.ipynb](HAM10000_baseline.ipynb) y ejecuta todas las celdas en orden.

Al finalizar, los resultados se guardan automáticamente en `experiments/<run_id>/`:

```
experiments/
└── 20260423_120000_real_only/
    ├── config.json             ← configuración y métricas del run
    ├── best_model.pt           ← checkpoint con mejor F1 en validación
    ├── history.json            ← métricas por época
    ├── test_metrics.json       ← recall, F1, AUC en test
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_curves.png
```
