---
name: "Verification Gates"
paths: ["tests/**/*", "scripts/data_processing/**/*", "scripts/training/**/*", "scripts/augmentation/**/*"]
priority: 50
---

## Verification Rules
- Check for lesion-level leakage before any training run.
- Validate that experiment metadata exists before and after execution.
- Ensure synthetic data generation logs model id, seed, prompt, and filtering results.
- Run the smallest relevant validation first, then the broader checks.
- Stop and report if a validation fails; do not silently continue.
