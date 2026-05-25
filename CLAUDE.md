# CLAUDE.md

## Project Identity
This repository supports a research project on binary skin lesion classification (Nevus vs Melanoma) using HAM10000 and synthetic augmentation with Stable Diffusion.

## Operating Modes
- PLAN: analyze, compare options, outline next steps, and validate assumptions without changing files unless explicitly requested.
- EXECUTE: make the smallest necessary change, keep artifacts reproducible, and log what was changed.
- VERIFY: run checks, inspect outputs, and confirm that the change did what it was supposed to do.

## Universal Standards
- Prefer reproducibility over convenience.
- Never hardcode dataset paths, seeds, model ids, or experiment names inside notebooks or scripts when a config file can hold them.
- Preserve the lesion-level split rule: split by `lesion_id`, not by image.
- Record dataset provenance for every run: source, DOI, version, release date, and download timestamp must appear in experiment metadata.
- Keep experiments traceable: every run should have a unique identifier, configuration snapshot, and metric summary.
- Do not overwrite experiment outputs unless the user explicitly asks for it.
- Treat synthetic data as experimental artifacts: store generation parameters, model version, and filtering decisions.

## Working Rules
- Make focused edits. Do not refactor unrelated code.
- Prefer scripts and config files for repeatable logic; use notebooks for exploration and reporting.
- Validate any change that affects data splits, augmentation, or metrics.
- If a task may change results or use external resources, pause and confirm before executing.

## Project Phases and Current State

### Phase 1 — Data & Split ✅ Done
- HAM10000 descargado desde Harvard Dataverse (DOI: `doi:10.7910/DVN/DBW86T`, versión 4)
- Split lesion-aware (70/15/15) por `lesion_id`, clases `nv` / `mel`
- Splits guardados en `data/processed/splits/` (train.csv, val.csv, test.csv + metadata.json)
- Scripts: `scripts/data_processing/01_extract.py`, `02_split.py`

### Phase 2 — Generación sintética ✅ Completada (imágenes en Drive)
- **Textual Inversion**: token `<mel-skin>` entrenado con 5 000 steps sobre ~800 imágenes de melanoma
  - Notebook: `notebooks/generation/HAM10000_textual_inversion.ipynb`
  - Embedding en Drive: `ham10000-augmentation/models/mel_skin_embedding_final.pt`
- **Generación SD** ejecutada en Colab T4 (`runwayml/stable-diffusion-v1-5`):
  - TI: ~4 500 imágenes — `notebooks/generation/HAM10000_generation.ipynb`
  - Img2Img: ~2 400 variaciones de reales
  - LoRA (rank=32): `notebooks/generation/HAM10000_lora_training.ipynb`
  - Derm-T2IM (img2img dermoscopy-specific, s=0.40): `notebooks/generation/HAM10000_derm_generation.ipynb`
- **WGAN-GP**: 100 epochs, imágenes 64×64 — `GAN/HAM10000_GAN.ipynb`, pesos en `GAN/checkpoints/generator_final.h5`
- Imágenes en Drive: `ham10000-augmentation/synthetic/{textual_inversion,img2img,lora,gan_final,derm_s040,derm_s005}/`

### Phase 3 — Evaluación de calidad sintética ✅ Completada
- Métricas FID e Inception Score por generador
- Script: `scripts/augmentation/evaluate_generation.py`
- Notebook: `HAM10000_quality_evaluation.ipynb`

### Phase 4 — Clasificación comparativa 🔄 En curso (Colab)
- Modelo: EfficientNet-B0 (15 epochs, lr=1e-4, AdamW, CosineAnnealingLR, seed=42)
- Checkpoint criterion: mejor val_loss
- Notebook canónico: `HAM10000_classification_comparative.ipynb`
- Diseño: comparar 5 generadores con volumen fijo 2× (N_REAL_MEL ≈ 801 sintéticas)
- 11 escenarios:
  | Escenario | Generador | Estado |
  |---|---|---|
  | `real_only` | — | ✅ AUC 0.911 / Recall 0.528 / F1 0.579 |
  | `real_2x_ti` | Textual Inversion | 🔄 |
  | `real_2x_lora` | LoRA SD v1.5 | 🔄 |
  | `real_2x_gan` | WGAN-GP | 🔄 |
  | `real_2x_derm` | Derm-T2IM s=0.40 | 🔄 |
  | `real_2x_derm005` | Derm-T2IM s=0.05 | 🔄 |
  | `synthetic_only_ti` | Textual Inversion | 🔄 |
  | `synthetic_only_lora` | LoRA SD v1.5 | 🔄 |
  | `synthetic_only_gan` | WGAN-GP | 🔄 |
  | `synthetic_only_derm` | Derm-T2IM s=0.05 | 🔄 |
  | `synthetic_only_derm040` | Derm-T2IM s=0.40 | 🔄 |
- Test set siempre 100% real (159 mel / 1 012 nv)
- Resultados Colab en Drive: `ham10000-augmentation/experiments/`

### Phase 5 — Webapp de generación ✅ Integrada
- Backend FastAPI + frontend HTML/JS: `webapp/`
- Modelo serializado: `webapp/backend/models/generator_final.h5` (WGAN-GP)
- Entorno separado: `conda create -n gan-tf python=3.10 tensorflow==2.15.0`
- Ver instrucciones: `webapp/backend/README.md`

## Convenciones de experimentos
- Cada run guarda: `config.json`, `test_metrics.json`, `history.json`, `best_model.pt`, curvas y matriz de confusión
- Nombre de carpeta: `YYYYMMDD_HHMMSS_<escenario>`
- No sobreescribir runs anteriores
