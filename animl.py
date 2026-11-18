"""Unified developer CLI for the AnimlVision monorepo.

Usage:
    python animl.py status
    python animl.py api [--port 8000]
    python animl.py web
    python animl.py art <subcommand...>
    python animl.py expansion compile
    python animl.py parks <manage|validate|analyze>
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _art_python() -> str:
    """Use the dedicated CUDA art-pipeline venv when present, otherwise current Python."""
    win_py = REPO_ROOT / "art-pipeline" / "lora" / "venv" / "Scripts" / "python.exe"
    posix_py = REPO_ROOT / "art-pipeline" / "lora" / "venv" / "bin" / "python"
    if win_py.exists():
        return str(win_py)
    if posix_py.exists():
        return str(posix_py)
    return sys.executable


def cmd_status(_args: argparse.Namespace) -> None:
    """Print CPU-only health & inventory across API, Web Frontend, Expansion catalog, and Art Pipeline."""
    api_cfg = REPO_ROOT / "api" / "species_config.json"
    exp_cfg = REPO_ROOT / "expansion" / "master_species_list.json"
    masters_dir = REPO_ROOT / "art-pipeline" / "masters"
    web_icons_dir = REPO_ROOT / "frontend" / "public" / "icons" / "web"
    lora_out = REPO_ROOT / "art-pipeline" / "lora" / "output"
    batches_dir = REPO_ROOT / "art-pipeline" / "configs" / "batches"
    runs_dir = REPO_ROOT / "art-pipeline" / "runs"

    api_count = len(json.loads(api_cfg.read_text(encoding="utf-8"))) if api_cfg.exists() else 0
    exp_summary = {}
    if exp_cfg.exists():
        exp_summary = json.loads(exp_cfg.read_text(encoding="utf-8")).get("summary", {})
    master_cnt = len([p for p in masters_dir.glob("*.png") if not p.name.startswith("icon-")]) if masters_dir.exists() else 0
    web_cnt = len(list(web_icons_dir.glob("*.webp"))) if web_icons_dir.exists() else 0
    v7_ready = (lora_out / "animl_style_v7-step00002000.safetensors").exists()
    v7_unet_ready = (lora_out / "animl_style_v7_unet_only.safetensors").exists()

    print("=== AnimlVision Monorepo Status (CPU-only) ===")
    print(f"API Species Config : {api_count} live species (api/species_config.json)")
    print(f"Expansion Catalog  : {exp_summary.get('total_species', 0)} species -> {exp_summary.get('by_class', {})}")
    print(f"Art Masters (PNG)  : {master_cnt} masters | Web Icons (WebP): {web_cnt}")
    print(f"v7 LoRA Checkpoints: full-2000={'OK' if v7_ready else 'missing'}, unet_only={'OK' if v7_unet_ready else 'missing'}")
    if batches_dir.exists():
        print(f"Batch Manifests    : {[p.name for p in sorted(batches_dir.glob('*.json'))]}")
    if runs_dir.exists():
        v7_runs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir() and ("v7" in d.name or "unseen" in d.name))
        print(f"Active v7 Runs     : {v7_runs}")


def cmd_api(args: argparse.Namespace) -> None:
    """Start the FastAPI backend server locally with hot-reload."""
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", args.host, "--port", str(args.port)]
    if not args.no_reload:
        cmd.append("--reload")
    subprocess.run(cmd, cwd=str(REPO_ROOT / "api"), check=True)


def cmd_web(args: argparse.Namespace) -> None:
    """Run Next.js frontend scripts (`dev`, `build`, `start`, or `lint`)."""
    npm_exe = "npm.cmd" if os.name == "nt" else "npm"
    subprocess.run([npm_exe, "run", args.script], cwd=str(REPO_ROOT / "frontend"), check=True)


def cmd_art(args: argparse.Namespace) -> None:
    """Forward commands to `art-pipeline/cli.py` using the CUDA venv when available."""
    cli_path = REPO_ROOT / "art-pipeline" / "cli.py"
    forward = args.art_args if args.art_args else ["status"]
    subprocess.run([_art_python(), str(cli_path), *forward], check=True)


def cmd_expansion(args: argparse.Namespace) -> None:
    """Run expansion catalog compilation or counting scripts."""
    scripts = {
        "compile": REPO_ROOT / "expansion" / "compile_master_list.py",
        "counts": REPO_ROOT / "expansion" / "current_100_counts.py",
        "test-batch": REPO_ROOT / "expansion" / "build_test_batch.py",
    }
    subprocess.run([sys.executable, str(scripts[args.action]), *args.extra], check=True)


def cmd_parks(args: argparse.Namespace) -> None:
    """Run national parks maintenance scripts in `scripts/parks/`."""
    scripts = {
        "manage": REPO_ROOT / "scripts" / "parks" / "manage_parks.py",
        "validate": REPO_ROOT / "scripts" / "parks" / "validate_nps.py",
        "analyze": REPO_ROOT / "scripts" / "parks" / "analyze_parks.py",
        "expand": REPO_ROOT / "scripts" / "parks" / "expand_parks_comprehensive.py",
    }
    subprocess.run([sys.executable, str(scripts[args.action]), *args.extra], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="animl.py", description="AnimlVision Monorepo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_st = sub.add_parser("status", help="Inspect monorepo state (CPU-only)")
    p_st.set_defaults(func=cmd_status)

    p_api = sub.add_parser("api", help="Run FastAPI backend server in api/")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--no-reload", action="store_true")
    p_api.set_defaults(func=cmd_api)

    p_web = sub.add_parser("web", help="Run Next.js frontend scripts in frontend/")
    p_web.add_argument("script", nargs="?", default="dev", choices=["dev", "build", "start", "lint"])
    p_web.set_defaults(func=cmd_web)

    p_art = sub.add_parser("art", help="Delegate to art-pipeline/cli.py (run `animl.py art --help` or `animl.py art status`)")
    p_art.add_argument("art_args", nargs=argparse.REMAINDER)
    p_art.set_defaults(func=cmd_art)

    p_exp = sub.add_parser("expansion", help="Run 1,777-species expansion scripts")
    p_exp.add_argument("action", choices=["compile", "counts", "test-batch"], default="compile")
    p_exp.add_argument("extra", nargs=argparse.REMAINDER)
    p_exp.set_defaults(func=cmd_expansion)

    p_parks = sub.add_parser("parks", help="Run National Parks scripts in scripts/parks/")
    p_parks.add_argument("action", choices=["manage", "validate", "analyze", "expand"], default="validate")
    p_parks.add_argument("extra", nargs=argparse.REMAINDER)
    p_parks.set_defaults(func=cmd_parks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
