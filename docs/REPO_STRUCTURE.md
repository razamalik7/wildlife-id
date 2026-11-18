# Repository Structure

```
aniML-vision/
├── animl.py                        # Top-level monorepo CLI
├── Dockerfile                      # Hugging Face Space API container definition
├── .github/workflows/huggingface.yml
├── README.md                       # Root overview + HF Spaces YAML header
│
├── frontend/                       # Next.js 16 web app + public WebP species icons
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── public/
│       ├── icons/web/              # 1024px WebP species illustrations
│       └── parks.json              # Validated park polygons/centers used by frontend & API
│
├── api/                            # FastAPI runtime (top-level stays minimal for Docker)
│   ├── main.py                     # FastAPI endpoints (/predict, /species, /parks/*)
│   ├── ai_engine.py                # EfficientNet-B3 + ConvNeXt late-fusion inference engine
│   ├── species_config.json         # Active runtime taxonomy & range config
│   ├── requirements.txt
│   ├── training/                   # PyTorch training scripts
│   ├── scraping/                   # iNaturalist dataset scraping & cleaning scripts
│   ├── evaluation/                 # Model confusion, accuracy & Oogway diagnostics
│   ├── taxonomy/                   # Taxon ID auditing, hierarchy & park cache scripts
│   └── reports/                    # Historical text & JSON evaluation logs
│
├── art-pipeline/                   # Automated SDXL v7 illustration pipeline
│   ├── cli.py                      # Art pipeline CLI (`status`, `run`, `swap`, `promote`, `export-web`, `normalize`)
│   ├── animl_art/                  # Modular v7 Python package (matte, sources, vetter, stylize, postprocess, runner)
│   ├── configs/
│   │   ├── palette.json
│   │   ├── references.csv
│   │   ├── prompts/framing_contract.txt
│   │   ├── batches/                # Versioned JSON species batch manifests (`unseen_10_batch1.json`, `unseen_10_batch2.json`)
│   │   └── benchmarks/             # Reference v7 metrics & vetted metadata baselines
│   ├── docs/                       # STYLE_SPEC.md, STYLE_BIBLE.md, PIPELINE_V7.md & research notes
│   ├── masters/                    # Accepted 2048px/1024px species master PNGs (source of truth)
│   ├── equalized_candidates_78/    # v3-equalized 78 style candidates (source for dataset_v7)
│   ├── reference-sheet/            # Frozen 6-species original style anchors
│   ├── lora/
│   │   ├── build_dataset_v7.py     # CPU-only 45-image dataset_v7 builder
│   │   ├── dataset_v7/             # Tracked 45-image v7 LoRA training set + captions
│   │   ├── dataset_v7.toml
│   │   ├── train_v7.ps1
│   │   ├── dataset/                # (gitignored) Legacy v6 25-image training set
│   │   ├── base/                   # (gitignored) SDXL 1.0 base checkpoint
│   │   ├── output/                 # (gitignored) Trained LoRA safetensors checkpoints
│   │   └── venv/                   # (gitignored) CUDA training & inference virtualenv
│   ├── vetting/                    # Local Flask curation UI (`serve.py`) & candidate metadata
│   ├── runs/                       # (gitignored) Per-batch working dirs (`cands_top3`, `cutouts`, `raw_stylized`, `eq_out`, `boards`)
│   └── archive/                    # Historical R&D scripts, style review sheets & logs
│       ├── experiments/
│       ├── raw_v1/
│       ├── style_review/
│       └── logs/
│
├── expansion/                      # 1,777-species expansion roster & scoping tools
├── scripts/                        # Repo-wide maintenance utilities
│   ├── parks/                      # NPS / Canadian park boundary & place_id scripts
│   └── taxonomy/                   # Standalone taxonomy audits
└── data/                           # Local SQLite DB & Supabase schema (`supabase_schema.sql`)
```
