"""AnimlVision modular v7 wildlife illustration pipeline package.

Keep heavyweight ML imports (`torch`, `diffusers`, `transformers`, `onnxruntime`)
lazy inside functions so CPU-only CLI commands (`status`, `--help`, manifest
building, metrics) run instantaneously without touching GPU VRAM.
"""

from pathlib import Path

__version__ = "7.0.0"

ART_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ART_ROOT.parent

LORA_DIR = ART_ROOT / "lora"
LORA_OUTPUT_DIR = LORA_DIR / "output"
SDXL_BASE_PATH = LORA_DIR / "base" / "sd_xl_base_1.0.safetensors"

MASTERS_DIR = ART_ROOT / "masters"
EQUALIZED_DIR = ART_ROOT / "equalized_candidates_78"
REFERENCE_SHEET_DIR = ART_ROOT / "reference-sheet"
RUNS_DIR = ART_ROOT / "runs"

CONFIGS_DIR = ART_ROOT / "configs"
BATCHES_DIR = CONFIGS_DIR / "batches"
BENCHMARKS_DIR = CONFIGS_DIR / "benchmarks"
DOCS_DIR = ART_ROOT / "docs"

EXPANSION_LIST_PATH = REPO_ROOT / "expansion" / "master_species_list.json"
WEB_ICONS_DIR = REPO_ROOT / "frontend" / "public" / "icons" / "web"

__all__ = [
    "__version__",
    "ART_ROOT",
    "REPO_ROOT",
    "LORA_DIR",
    "LORA_OUTPUT_DIR",
    "SDXL_BASE_PATH",
    "MASTERS_DIR",
    "EQUALIZED_DIR",
    "REFERENCE_SHEET_DIR",
    "RUNS_DIR",
    "CONFIGS_DIR",
    "BATCHES_DIR",
    "BENCHMARKS_DIR",
    "DOCS_DIR",
    "EXPANSION_LIST_PATH",
    "WEB_ICONS_DIR",
]
