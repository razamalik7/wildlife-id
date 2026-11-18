---
title: aniML vision
emoji: 🐻
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# AnimlVision (`wildlife-id`)

Full-stack wildlife identification monorepo combining a **Next.js 16 Anidex web app**, a **PyTorch hierarchical vision ensemble API (FastAPI on Hugging Face Spaces)**, a **1,777-species North American expansion catalog**, and an automated **SDXL v7 wildlife illustration pipeline**.

## Live Deployments
- **Frontend (Vercel):** [https://wildlife-id.vercel.app](https://wildlife-id.vercel.app)
- **Inference API (Hugging Face Spaces):** [https://huggingface.co/spaces/raza44/aniML](https://huggingface.co/spaces/raza44/aniML)

## Repository Map

| Directory / File | Role | Guide |
|---|---|---|
| `animl.py` | Top-level developer CLI (`status`, `api`, `web`, `art`, `expansion`, `parks`) | — |
| `frontend/` | Next.js 16 + Tailwind CSS v4 + Leaflet Anidex, camera ID & park maps | [package.json](frontend/package.json) |
| `api/` | Production FastAPI backend + EfficientNet-B3 / ConvNeXt late-fusion inference | [api/README.md](api/README.md) |
| `art-pipeline/` | Automated `v7` LoRA + MultiControlNet illustration & `v3` outline pipeline | [art-pipeline/README.md](art-pipeline/README.md) |
| `expansion/` | 1,777-species expansion roster, scoping logic & observation counts | [expansion/README.md](expansion/README.md) |
| `scripts/` | Offline park ID, NPS validation & taxonomy maintenance scripts | [scripts/README.md](scripts/README.md) |
| `data/` | SQLite / Supabase schemas & local reference datasets | — |
| `docs/` | System architecture & full directory structure documentation | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| `Dockerfile` | Hugging Face Space Docker runtime (copies only `api/` runtime + `parks.json`) | [Dockerfile](Dockerfile) |

## Quickstart

Check overall repo health (CPU-only):
```bash
python animl.py status
```

Run the backend API (`http://127.0.0.1:8000`) and frontend (`http://localhost:3000`):
```bash
python animl.py api
python animl.py web
```

Check art pipeline status or launch an unseen species illustration batch:
```bash
python animl.py art status
python animl.py art run --batch unseen_batch2 --manifest art-pipeline/configs/batches/unseen_10_batch2.json --phase all
```

## License
[MIT](LICENSE)
