---
name: "Notebook Workflow"
scope: "/**/*.ipynb"
priority: 20
apply_to: ["ipynb"]
---

## Notebook Rules
- Use notebooks for EDA, explanation, and result presentation.
- Keep execution linear and restart kernels before validation runs.
- Do not hide critical logic in notebook state; move reusable logic to scripts.
- Save important figures and tables to `reports/` or another tracked artifact location.
- If a notebook depends on data paths or seeds, load them from config instead of hardcoding.
