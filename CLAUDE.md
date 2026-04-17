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
- Keep experiments traceable: every run should have a unique identifier, configuration snapshot, and metric summary.
- Do not overwrite experiment outputs unless the user explicitly asks for it.
- Treat synthetic data as experimental artifacts: store generation parameters, model version, and filtering decisions.

## Working Rules
- Make focused edits. Do not refactor unrelated code.
- Prefer scripts and config files for repeatable logic; use notebooks for exploration and reporting.
- Validate any change that affects data splits, augmentation, or metrics.
- If a task may change results or use external resources, pause and confirm before executing.

## Recommended First Milestone
1. Define the project config and experiment registry.
2. Establish the lesion-aware split pipeline.
3. Create the augmentation workflow and quality checks.
4. Add the classifier training and reporting flow.
