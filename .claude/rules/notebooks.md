---
name: "Notebook Workflow"
paths: ["**/*.ipynb"]
priority: 20
---

## Notebook Rules
- Use notebooks for EDA, explanation, and result presentation.
- Keep execution linear and restart kernels before validation runs.
- Do not hide critical logic in notebook state; move reusable logic to scripts.
- Save experiment figures (curves, confusion matrix, ROC) inside the run directory (`experiments/<run_id>/`). Save final presentation artifacts to `reports/`.
- If a notebook depends on data paths or seeds, load them from config instead of hardcoding.
