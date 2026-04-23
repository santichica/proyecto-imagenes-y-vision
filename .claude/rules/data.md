---
name: "Data Preparation"
paths: ["scripts/data/**/*", "scripts/augmentation/**/*", "config/**/*"]
priority: 30
---

## Data Rules
- Split by `lesion_id` to prevent leakage.
- Keep train/validation/test boundaries fixed once validated.
- Record dataset version, source paths, and seed in every run.
- Log augmentation settings and synthetic sample provenance.
- Prefer small, auditable preprocessing steps over opaque notebooks.
