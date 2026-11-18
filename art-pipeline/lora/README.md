# SDXL Style LoRA (`animl_style_v7`)

Houses the SDXL style-LoRA training datasets, kohya configs, and PowerShell runners.

## Contents
- `dataset_v7/` *(tracked in git)* — Curated 45-image training set (`img_6x`, `img_4x`, `img_2x`) built from `art-pipeline/equalized_candidates_78/`.
- `build_dataset_v7.py` — Deterministic CPU-only dataset builder producing `dataset_v7/`.
- `dataset_v7.toml` / `train_v7.ps1` / `sample_prompts_v7.txt` — Rank-32, 2,000-step SDXL LoRA training configuration.
- `build_dataset.py` / `dataset.toml` / `train.ps1` — Legacy v3–v6 configuration.
- *(Gitignored local artifacts)* `base/` (`sd_xl_base_1.0.safetensors`), `output/` (`animl_style_v7-step00002000.safetensors`, `animl_style_v7_unet_only.safetensors`), `dataset/` (legacy v6), `sd-scripts/`, `venv/`, `eval_refs/`.

> Note: Keep `dataset/` (legacy v6) and `dataset_v7/` untouched so trained checkpoints remain reproducible.
