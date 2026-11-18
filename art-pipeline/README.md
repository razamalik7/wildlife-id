# AnimlVision Art Pipeline (v7)

Automated style-locked wildlife illustration system generating 1024x1024 cel-shaded species icons using **SDXL + `animl_style_v7` LoRA + Canny/Depth MultiControlNet**, followed by **BiRefNet matting** and the **`v3` smooth curvature-tapered outline equalizer**.

## Directory Layout
| Path | Role |
|---|---|
| `cli.py` | Unified pipeline CLI (`status`, `prefetch`, `run`, `swap`, `review`, `apply-review`, `promote`, `export-web`, `normalize`) |
| `animl_art/` | Modular core library (`matte`, `sources`, `vetter`, `stylize`, `postprocess`, `prefetch`, `review_server`, `runner`) |
| `configs/` | `palette.json`, `references.csv`, `prompts/`, disk caches (`cache/`), batch manifests (`batches/`), baseline benchmarks (`benchmarks/`) |
| `docs/` | [STYLE_SPEC.md](docs/STYLE_SPEC.md), [STYLE_BIBLE.md](docs/STYLE_BIBLE.md), [PIPELINE_V7.md](docs/PIPELINE_V7.md), research notes |
| `masters/` | Accepted canonical species PNGs (source of truth for frontend icons) |
| `equalized_candidates_78/` | 78 `v3`-equalized style candidates (`dataset_v7` built from 45 picks) |
| `reference-sheet/` | Frozen 6-species original style anchors |
| `lora/` | SDXL LoRA training scripts, tracked `dataset_v7/` (45 images), local weights/venv ([lora/README.md](lora/README.md)) |
| `vetting/` | Local Flask photo curation UI (`serve.py`) & candidate metadata |
| `runs/` | *(gitignored)* Per-batch artifacts (`cands_top3`, `cutouts_final`, `raw_stylized`, `eq_out`, `boards`) |
| `archive/` | Historical R&D experiment scripts, early `raw_v1/`, style review sheets & logs |

## Common Workflows (via Repo Root CLI)

Check pipeline state & LoRA readiness (CPU-only):
```bash
python animl.py art status
```

Run a manifest end-to-end (automatically spawns isolated subprocesses for `1a -> 1b -> stylize` so 16 GB VRAM stays clean):
```bash
python animl.py art run --batch unseen_batch2 --manifest art-pipeline/configs/batches/unseen_10_batch2.json --phase all
```

Auto-sample 10 brand-new unseen species from `expansion/master_species_list.json` and run all phases:
```bash
python animl.py art run --batch auto_10 --sample-unseen 10 --phase all
```

Swap an individual candidate photo (`rank 0..3`) and re-stylize just that species:
```bash
python animl.py art swap --batch unseen_batch2 10_whitetip_reef_shark 1
python animl.py art run --batch unseen_batch2 --phase stylize --force 10_whitetip_reef_shark
```

Promote approved `eq_out/*.png` illustrations into `masters/` and regenerate frontend WebP icons:
```bash
python animl.py art promote --batch unseen_batch2 --export-web
```

## Scaling to the Top 500 Unseen iNaturalist Species

Two pre-built manifests live under `configs/batches/` (excluding all training-set and `masters/` species):
- **`top500_inat_balanced.json`** *(default)* — Top `240 Aves + 90 Mammalia + 90 Reptilia + 55 Amphibia + 25 Elasmobranchii` by iNaturalist observation count, also split into 10×50-species tiers under `configs/batches/top500_tiers/tier_01.json` .. `tier_10.json`.
- **`top500_inat_raw.json`** — Pure unstratified #1–#500 most-observed species on iNaturalist.

Pre-warm iNaturalist & Wikipedia caches on CPU (`configs/cache/`):
```bash
python animl.py art prefetch --build-top500
```

Run the Top 500 in 50-species tiers into a shared batch directory (fault-isolated, incremental caching, automatic QA flags):
```bash
python animl.py art run --batch top500 --manifest art-pipeline/configs/batches/top500_inat_balanced.json --offset 0 --limit 50 --phase all
python animl.py art run --batch top500 --manifest art-pipeline/configs/batches/top500_inat_balanced.json --offset 50 --limit 50 --phase all
```

Inspect QA flags (`low_black_outline`, `fur_halo_gap`, `low_aa`, `low_pose_score`, `repair:*`) and queue candidate swaps in the local browser review UI, then apply swaps and promote approved masters:
```bash
python animl.py art review --batch top500
python animl.py art apply-review --batch top500 --promote --export-web
```
