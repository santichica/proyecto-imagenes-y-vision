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
  - Notebook: `HAM10000_textual_inversion.ipynb`
  - Embedding en Drive: `models/mel_skin_embedding_final.pt`
- **Generación** ejecutada en Colab T4 (`runwayml/stable-diffusion-v1-5`):
  - TI: 4 500 imágenes sintéticas
  - Img2Img: ~2 400 variaciones de reales
  - Notebook: `HAM10000_generation.ipynb`
- **Pendiente**: descargar `synthetic/` de Drive → `data/synthetic/` (decidir si se entrena en Colab o local primero)

### Phase 3 — Filtrado de calidad sintética ⬜ Pendiente
- Filtro piel/no-piel sobre las imágenes generadas
- Métricas FID / IS para cuantificar calidad
- Guardar parámetros de filtrado en metadata

### Phase 4 — Clasificación comparativa ⬜ Pendiente
- Modelo: EfficientNet-B0 (15 epochs, lr=1e-4, CosineAnnealingLR, WeightedRandomSampler)
- Notebook: `HAM10000_classification.ipynb` (Colab + local, resumible por sesión)
- Script de preparación de datos: `scripts/augmentation/prepare_for_colab_classification.py`
- Escenarios:
  | Escenario | Train mel | Estado |
  |---|---|---|
  | `real_only` | ~800 reales | ✅ baseline local: AUC 0.926 / Recall 0.843 / F1 0.604 |
  | `real_balanced` | 800 real + ~3900 sint. (mel ≈ nv) | ⬜ pendiente |
  | `real_2x` | 800 real + 800 sint. (dobla minoría) | ⬜ pendiente |
  | `synthetic_only` | ~800 sint. (reemplaza real mel) | ⬜ pendiente |
- El test set es siempre solo imágenes reales
- Resultados del baseline local: `experiments/20260426_204545_real_only/`
- Resultados de Colab irán a Drive: `experiments/<scenario>/`

### Exploración paralela (rama `gabriel_develop`)
- WGAN-GP en TensorFlow/Keras, imágenes 64×64, 100 epochs completados
- No persiste pesos ni imágenes generadas — requiere reentrenar si se usa
- Entorno separado: `conda create -n gan-tf python=3.10 tensorflow==2.15.0`

## Convenciones de experimentos
- Cada run guarda: `config.json`, `test_metrics.json`, `history.json`, `best_model.pt`, curvas y matriz de confusión
- Nombre de carpeta: `YYYYMMDD_HHMMSS_<escenario>`
- No sobreescribir runs anteriores
