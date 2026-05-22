# Metodología — Marco CRISP-ML(Q)

Este proyecto sigue el proceso iterativo definido por **CRISP-ML(Q)** (Studer et al., 2021), un marco para el desarrollo de sistemas de Machine Learning con énfasis en calidad y reproducibilidad. A continuación se describe cómo cada fase se materializó en artefactos concretos del repositorio.

> **Referencia:** Studer, L. et al. (2021). CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology. *Machine Learning and Knowledge Extraction*, 3(2), 392–413. https://doi.org/10.3390/make3020020

---

## Tabla de mapeo CRISP-ML(Q) → Artefactos del proyecto

| Fase CRISP-ML(Q) | Descripción en este proyecto | Artefactos |
|---|---|---|
| **1. Business & Research Understanding** | Definir la pregunta de investigación: ¿puede la augmentación sintética con Stable Diffusion mejorar la detección de melanoma en un dataset desbalanceado? | `CLAUDE.md` §Fases, `config/project.yaml` |
| **2. Data Understanding** | Análisis exploratorio de HAM10000: distribución de clases, diversidad morfológica, desbalance severo mel/nv | `notebooks/exploration/HAM10000_EDA.ipynb`, `data/processed/splits/metadata.json` |
| **3. Data Preparation** | Split lesion-aware (70/15/15 por `lesion_id`), extracción de imágenes, preparación de conjuntos para augmentación y clasificación | `scripts/data_processing/01_extract.py`, `02_split.py`, `tests/test_split_leakage.py` |
| **4. Modeling** | Entrenamiento de modelos generativos (TI, img2img, LoRA, Derm-T2IM, WGAN-GP) y clasificador EfficientNet-B0 bajo 4 escenarios de augmentación | `notebooks/generation/`, `GAN/HAM10000_GAN.ipynb`, `HAM10000_classification_final.ipynb` |
| **5. Evaluation** | Comparación de escenarios con métricas clínicas (AUC, Recall melanoma, F1) y de calidad generativa (FID, Inception Score) | `HAM10000_quality_evaluation.ipynb`, `scripts/augmentation/evaluate_generation.py`, `experiments/` |
| **6. Deployment** | Aplicación web para generación interactiva de imágenes dermoscópicas con el modelo GAN entrenado | `webapp/`, `GAN/checkpoints/generator_final.h5` |

---

## 1. Business & Research Understanding

### Motivación clínica

El melanoma representa menos del 5% de los cánceres de piel pero causa más del 75% de las muertes relacionadas. La detección temprana es crítica y los sistemas de asistencia diagnóstica basados en imágenes dermoscópicas han demostrado rendimiento comparable al del dermatólogo experto. Sin embargo, el desbalance de clases en los datasets disponibles (HAM10000: ~6,700 nevus vs ~1,113 melanoma en imágenes totales) representa un reto metodológico central.

### Pregunta de investigación

> ¿Puede la augmentación sintética con modelos generativos (Stable Diffusion, WGAN-GP) mejorar la sensibilidad al melanoma de un clasificador EfficientNet-B0 respecto a un baseline entrenado solo con datos reales?

### Métricas de éxito

La métrica primaria es el **Recall de melanoma** (sensibilidad): priorizar la detección del positivo real sobre la precisión, dado el costo asimétrico de un falso negativo clínico. Las métricas secundarias son AUC-ROC y F1 de melanoma.

---

## 2. Data Understanding

### Dataset: HAM10000

- **Fuente:** Harvard Dataverse, DOI: [`10.7910/DVN/DBW86T`](https://doi.org/10.7910/DVN/DBW86T), versión 4
- **Contenido:** 10,015 imágenes dermoscópicas de 7 tipos de lesiones
- **Subconjunto utilizado:** clases binarias `mel` (melanoma) y `nv` (nevus melanocítico)
  - Melanoma: 1,113 imágenes únicas / ~801 en train
  - Nevus: 6,705 imágenes únicas / ~4,694 en train
- **Ratio de desbalance train:** ~1:6 (mel:nv)

### Hallazgos del EDA

- Múltiples imágenes por lesión (`lesion_id`): ignorar esto en el split introduce data leakage.
- Alta variabilidad intraclase en melanoma (pigmentación, bordes, tamaño).
- Las imágenes de nevus dominan en volumen y en diversidad visual.

Ver análisis completo: [`HAM10000_EDA.ipynb`](notebooks/exploration/HAM10000_EDA.ipynb)

---

## 3. Data Preparation

### Split lesion-aware

El split se realiza a nivel de `lesion_id`, **no** a nivel de imagen, para evitar que variaciones de la misma lesión aparezcan en train y test simultáneamente.

| Conjunto | % de lesiones | Imágenes mel | Imágenes nv |
|---|---|---|---|
| Train | 70% | ~801 | ~4,694 |
| Val | 15% | ~171 | ~1,005 |
| Test | 15% | ~141 | ~1,006 |

- Script: `scripts/data_processing/02_split.py`
- Verificación de leakage: `tests/test_split_leakage.py`
- Metadatos del split: `data/processed/splits/metadata.json`

El **test set siempre contiene solo imágenes reales**, independientemente del escenario de augmentación en train.

### Preparación para augmentación sintética

```bash
python scripts/augmentation/prepare_for_colab.py          # para TI / img2img
python scripts/augmentation/prepare_for_colab_classification.py  # para clasificación
```

---

## 4. Modeling

### 4.1 Modelos generativos

Se exploraron cuatro estrategias de síntesis de imágenes dermoscópicas de melanoma:

#### Textual Inversion (TI)
- **Base:** `runwayml/stable-diffusion-v1-5`
- **Token aprendido:** `<mel-skin>` (5,000 pasos, ~801 imágenes de melanoma)
- **Imágenes generadas:** 4,500
- **Notebook:** `notebooks/generation/HAM10000_textual_inversion.ipynb` → `notebooks/generation/HAM10000_generation.ipynb`

#### Img2Img
- Variaciones de imágenes reales de melanoma con `strength` configurable
- **Imágenes generadas:** 2,403
- **Notebook:** `notebooks/generation/HAM10000_generation.ipynb`

#### LoRA Fine-tuning
- Adaptadores de bajo rango sobre SD v1.5, entrenados sobre melanoma
- Permite control más fino sobre características dermoscópicas específicas
- **Notebook:** `notebooks/generation/HAM10000_lora_training.ipynb`

#### Derm-T2IM (img2img dermatología-específico)
- Modelo base especializado en imágenes dermoscópicas
- Generación condicionada por imagen real (img2img)
- **Notebook:** `notebooks/generation/HAM10000_derm_generation.ipynb`

#### WGAN-GP (GAN clásica)
- Framework: TensorFlow/Keras, imágenes 64×64 px
- Entrenamiento: 100 epochs, ~800 imágenes de melanoma
- **Notebook:** `GAN/HAM10000_GAN.ipynb`
- **Checkpoint:** `GAN/checkpoints/generator_final.h5`

### 4.2 Modelo de clasificación

- **Arquitectura:** EfficientNet-B0 preentrenado (ImageNet)
- **Hiperparámetros fijos en todos los escenarios:**
  - Epochs: 15 | LR: 1e-4 | Scheduler: CosineAnnealingLR
  - Batch size: 32 | Optimizer: Adam | Semilla: 42
  - WeightedRandomSampler para compensar desbalance residual
- **Notebook:** `HAM10000_classification_comparative.ipynb`

### 4.3 Escenarios experimentales

El diseño compara cuatro métodos generativos bajo el mismo volumen de augmentación (**2×**: tantas sintéticas como reales de melanoma, ~801). Esto hace los resultados comparables entre generadores sin confundir el efecto del volumen con el del método. El escenario `synthetic_only_ti` se incluye como ablación para evaluar si las imágenes sintéticas pueden sustituir completamente a las reales, siguiendo el diseño de Akrout et al. (2023).

| Escenario | Train melanoma | Método generativo | Pregunta central |
|---|---|---|---|
| `real_only` | 801 reales | — | Baseline sin augmentación |
| `real_2x_ti` | 801 real + 801 TI | Textual Inversion (SD v1.5) | ¿TI mejora el Recall de melanoma? |
| `real_2x_lora` | 801 real + 801 LoRA | LoRA fine-tuning (SD v1.5) | ¿LoRA supera a TI? |
| `real_2x_gan` | 801 real + 801 GAN | WGAN-GP 64×64 px | ¿Una GAN clásica es competitiva con SD? |
| `real_2x_derm` | 801 real + 801 Derm | Derm-T2IM img2img (s=0.40) | ¿Un modelo dermoscopy-specific reduce el distributional shift? |
| `synthetic_only_ti` | 801 TI (sin reales) | Textual Inversion | ¿Las sintéticas pueden sustituir a las reales? |

**Control de cantidad:** todos los escenarios `real_2x_*` usan exactamente `N_REAL_MEL` ≈ 801 imágenes sintéticas. El test set es siempre 100% real (159 mel / 1,012 nv).

> Diseño inspirado en: Akrout et al. (2023). *Diffusion-based Data Augmentation for Skin Disease Classification*. arXiv:2301.04802

---

## 5. Evaluation

### 5.1 Métricas de clasificación

| Métrica | Justificación |
|---|---|
| Recall melanoma | Métrica primaria; minimiza falsos negativos (consecuencia clínica alta) |
| AUC-ROC | Rendimiento global independiente del umbral de decisión |
| F1 melanoma | Balance precision-recall en la clase minoritaria |
| Accuracy | Referencia general; puede ser engañosa con desbalance |

### 5.2 Métricas de calidad generativa

| Métrica | Descripción |
|---|---|
| FID (Fréchet Inception Distance) | Distancia entre distribuciones de imágenes reales y sintéticas; menor es mejor |
| Inception Score (IS) | Calidad y diversidad de las imágenes generadas; mayor es mejor |

Script de evaluación: `scripts/augmentation/evaluate_generation.py`  
Notebook de evaluación: `HAM10000_quality_evaluation.ipynb`

### 5.3 Resultados baseline (`real_only`)

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.926 |
| Recall melanoma | 0.843 |
| F1 melanoma | 0.604 |
| Precisión melanoma | 0.470 |
| Accuracy | 0.850 |

Experimento: `experiments/20260426_204545_real_only/`

---

## 6. Deployment

### Aplicación web

La webapp permite generar imágenes dermoscópicas sintéticas de melanoma a través de una interfaz gráfica, usando el generador WGAN-GP entrenado.

- **Backend:** FastAPI (`webapp/backend/main.py`), carga `generator_final.h5`
- **Frontend:** HTML/CSS/JS (`webapp/front/`)
- **Modelo serializado:** `webapp/backend/models/generator_final.h5`

Ver instrucciones de ejecución: `webapp/backend/README.md`

---

## Proceso iterativo

CRISP-ML(Q) es explícitamente cíclico. En este proyecto se observaron las siguientes iteraciones:

```
Data Understanding
      ↓
  Detectar desbalance severo mel/nv
      ↓
  Modelado generativo (TI → img2img → LoRA → Derm-T2IM → GAN)
      ↓
  Evaluación de calidad sintética (FID/IS)  ←──────┐
      ↓                                             │
  Clasificación comparativa (4 escenarios)          │
      ↓                                             │
  ¿Mejora el Recall mel?  ──── No ─────────────────┘
      ↓ Sí
  Análisis de resultados y conclusiones
```

Cada generador se evaluó primero con FID/IS antes de incorporarlo al pipeline de clasificación, evitando el costo computacional de entrenar el clasificador con imágenes de baja calidad.
