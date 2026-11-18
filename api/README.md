# AnimlVision Inference API

Production FastAPI backend serving species identification (`EfficientNet-B3 + ConvNeXt-Tiny` late-fusion ensemble) and park sightings proxies.

## Runtime Files (Top-Level Only — Used by Docker & Hugging Face Spaces)
- `main.py` — FastAPI server (`/`, `/species`, `/predict`, `/parks/observations`, `/inat/*`, `/parks/recent-sightings`)
- `ai_engine.py` — Late-fusion ensemble (`WildlifeLateFusion`), hierarchy heads, range checks & expert overrides
- `species_config.json` — Active species metadata & taxonomy configuration
- `requirements.txt` — Minimal runtime dependencies
- `oogway_b3_best.pth`, `oogway_convnext_final.pth` — Local weights (gitignored; downloaded by [Dockerfile](../Dockerfile) during HF build)

## Local Run
From the repo root:
```bash
python animl.py api
```
Or directly from `api/`:
```bash
python -m uvicorn main:app --reload --port 8000
```

## Organized Subdirectories (Offline Tools & History)
| Folder | Contents |
|---|---|
| `training/` | PyTorch ensemble & Oogway training/fine-tuning scripts |
| `scraping/` | iNaturalist dataset scrapers, blur/crop filters & balancers |
| `evaluation/` | Confusion matrices, per-class metrics & diagnostic harnesses |
| `taxonomy/` | Taxon ID verification, hierarchy builders & park cache tools |
| `reports/` | Historical training, validation & audit output reports |
