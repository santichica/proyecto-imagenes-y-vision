---
name: "Training and Evaluation"
scope: ["/scripts/training/**", "/experiments/**"]
priority: 40
apply_to: ["py", "json", "md"]
---

## Model Rules
- Keep every training run reproducible and uniquely identified.
- Store the model config, seed, data split reference, and metrics with the experiment.
- Compare scenarios consistently: real-only, hybrid, and synthetic-only.
- Favor downstream utility metrics for the task: recall, F1, and AUC.
- Do not hide evaluation logic inside ad hoc notebook cells.
