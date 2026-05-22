# Notebooks

> **Importante:** lanzar siempre Jupyter desde la raíz del proyecto para que `Path.cwd()` resuelva correctamente.
> ```bash
> cd /ruta/al/proyecto
> jupyter lab
> ```

## Estructura

### `/` (raíz del proyecto) — Notebooks canónicos

| Notebook | Descripción |
|---|---|
| `HAM10000_classification_final.ipynb` | Clasificación comparativa: escenarios TI, LoRA, GAN, mezclas |
| `HAM10000_quality_evaluation.ipynb` | Evaluación de calidad sintética: FID e Inception Score |

### `generation/` — Entrenamiento y generación de imágenes sintéticas

| Notebook | Descripción |
|---|---|
| `HAM10000_textual_inversion.ipynb` | Entrena el token `<mel-skin>` (5 000 steps, SD v1.5) |
| `HAM10000_generation.ipynb` | Genera ~4 500 TI + ~2 400 img2img |
| `HAM10000_lora_training.ipynb` | Fine-tuning LoRA sobre SD v1.5 para melanoma |
| `HAM10000_derm_generation.ipynb` | Generación con Derm-T2IM (img2img dermatología-específico) |

Ver también: `GAN/HAM10000_GAN.ipynb` — entrenamiento WGAN-GP (TensorFlow).

### `exploration/` — Análisis exploratorio y baseline

| Notebook | Descripción |
|---|---|
| `HAM10000_EDA.ipynb` | Análisis exploratorio: distribución de clases, visualizaciones |
| `HAM10000_baseline.ipynb` | Baseline EfficientNet-B0 en datos reales (experimento inicial) |

### `experiments/` — Variantes experimentales de clasificación

| Notebook | Descripción |
|---|---|
| `HAM10000_classification.ipynb` | Los 4 escenarios base (real_only, real_2x, real_balanced, synthetic_only) |
| `HAM10000_classification_finetune.ipynb` | Estrategia de entrenamiento en dos etapas para reducir distributional shift |
| `HAM10000_classification_isolated.ipynb` | Evaluación aislada por fuente sintética (img2img, TI, LoRA, GAN) |
