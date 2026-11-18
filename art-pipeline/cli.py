"""Unified CLI for the AnimlVision v7 wildlife illustration pipeline.

Examples:
    python art-pipeline/cli.py status
    python art-pipeline/cli.py run --batch unseen_batch2 --manifest art-pipeline/configs/batches/unseen_10_batch2.json --phase all
    python art-pipeline/cli.py swap --batch unseen_batch2 10_whitetip_reef_shark 1
    python art-pipeline/cli.py promote --batch unseen_batch2 --export-web
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animl_art import (
    ART_ROOT,
    BATCHES_DIR,
    EXPANSION_LIST_PATH,
    LORA_OUTPUT_DIR,
    MASTERS_DIR,
    RUNS_DIR,
    WEB_ICONS_DIR,
)


def cmd_status(_args: argparse.Namespace) -> None:
    """CPU-only status report of masters, 1,777 expansion coverage, LoRAs, and active runs."""
    masters = sorted(p for p in MASTERS_DIR.glob("*.png") if not p.name.startswith("icon-")) if MASTERS_DIR.exists() else []
    web_icons = sorted(WEB_ICONS_DIR.glob("*.webp")) if WEB_ICONS_DIR.exists() else []
    loras = sorted(LORA_OUTPUT_DIR.glob("*.safetensors")) if LORA_OUTPUT_DIR.exists() else []

    print("=== AnimlVision Art Pipeline (v7) ===")
    print(f"Masters PNGs   : {len(masters)} ({MASTERS_DIR})")
    print(f"WebP Icons     : {len(web_icons)} ({WEB_ICONS_DIR})")
    print(f"LoRA Weights   : {len(loras)} checkpoints under {LORA_OUTPUT_DIR}")
    for lw in loras:
        if "v7" in lw.name and ("2000" in lw.name or lw.name in ("animl_style_v7.safetensors", "animl_style_v7_unet_only.safetensors")):
            print(f"  - {lw.name} ({lw.stat().st_size / (1024 * 1024):.1f} MB)")

    if EXPANSION_LIST_PATH.exists():
        catalog = json.loads(EXPANSION_LIST_PATH.read_text(encoding="utf-8"))
        summary = catalog.get("summary", {})
        print(f"Expansion List : {summary.get('total_species', len(catalog.get('species', [])))} species -> {summary.get('by_class', {})}")

    if BATCHES_DIR.exists():
        batches = sorted(BATCHES_DIR.glob("*.json"))
        print(f"Batch Manifests: {[b.name for b in batches]}")

    if RUNS_DIR.exists():
        runs = sorted(d.name for d in RUNS_DIR.iterdir() if d.is_dir())
        v7_runs = [r for r in runs if "v7" in r or "unseen" in r]
        print(f"Local Runs     : {len(runs)} total dirs under {RUNS_DIR} (v7/unseen: {len(v7_runs)})")
        for rname in v7_runs:
            eq_cnt = len(list((RUNS_DIR / rname / "eq_out").glob("*.png"))) if (RUNS_DIR / rname / "eq_out").exists() else 0
            print(f"  - {rname} (eq_out={eq_cnt})")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute one or all phases of the v7 art pipeline for a batch."""
    from animl_art.runner import (
        load_species_specs,
        run_phase_1a,
        run_phase_1b,
        run_phase_all_isolated,
        run_phase_stylize,
    )

    work_dir = RUNS_DIR / args.batch
    manifest_p = Path(args.manifest) if args.manifest else None
    boards_p = Path(args.boards_out) if args.boards_out else None

    if args.phase == "all":
        run_phase_all_isolated(
            cli_path=Path(__file__).resolve(),
            batch_name=args.batch,
            manifest_path=manifest_p,
            species_queries=args.species,
            sample_unseen=args.sample_unseen,
            force_stems=args.force,
            boards_dir=boards_p,
        )
        return

    species = load_species_specs(
        work_dir,
        manifest_path=manifest_p,
        species_queries=args.species,
        sample_unseen=args.sample_unseen,
    )
    if args.phase == "1a":
        run_phase_1a(work_dir, species, force_stems=set(args.force))
    elif args.phase == "1b":
        run_phase_1b(work_dir, species, force_stems=set(args.force), boards_dir=boards_p)
    elif args.phase == "stylize":
        run_phase_stylize(
            work_dir,
            species,
            force_stems=set(args.force),
            unet_only_stems=set(args.unet_only),
            boards_dir=boards_p,
        )


def cmd_swap(args: argparse.Namespace) -> None:
    from animl_art.runner import swap_candidate

    swap_candidate(RUNS_DIR / args.batch, args.stem, args.cand_rank)


def cmd_promote(args: argparse.Namespace) -> None:
    from animl_art.runner import promote_to_masters

    count = promote_to_masters(RUNS_DIR / args.batch, stems=args.stems or None, export_web=args.export_web)
    print(f"Promoted {count} species into {MASTERS_DIR}")


def cmd_export_web(args: argparse.Namespace) -> None:
    from animl_art.postprocess import export_web_icons

    export_web_icons(Path(args.icons), Path(args.out), size=args.size, quality=args.quality)


def cmd_normalize(args: argparse.Namespace) -> None:
    from animl_art.matte import normalize_canvas

    src, out_dir = Path(args.src), Path(args.out)
    pngs = sorted(src.glob("*.png")) if src.is_dir() else [src]
    if not pngs:
        sys.exit(f"no .png files under {src}")
    if out_dir.resolve() == (src if src.is_dir() else src.parent).resolve():
        sys.exit("refusing to write into the input directory; pick a different --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    for png in pngs:
        print(normalize_canvas(png, out_dir, canvas=args.canvas, fill=args.fill))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="art-pipeline/cli.py", description="AnimlVision v7 Art Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="CPU-only pipeline status, masters, LoRA weights, and runs")
    p_status.set_defaults(func=cmd_status)

    p_run = sub.add_parser("run", help="Run a batch through v7 pipeline stages")
    p_run.add_argument("--batch", required=True, help="Run directory name under art-pipeline/runs/<batch>")
    p_run.add_argument("--manifest", help="Path to JSON list of species entries")
    p_run.add_argument("--species", nargs="+", help="One or more species common/scientific/slug names")
    p_run.add_argument("--sample-unseen", type=int, default=0, help="Auto-sample N strictly unseen species")
    p_run.add_argument("--phase", choices=["1a", "1b", "stylize", "all"], default="all")
    p_run.add_argument("--force", nargs="*", default=[], help="Force recompute specific species stems")
    p_run.add_argument("--unet-only", nargs="*", default=[], help="Force unet_only LoRA weights for stems")
    p_run.add_argument("--boards-out", help="Optional directory to also write contact sheet boards")
    p_run.set_defaults(func=cmd_run)

    p_swap = sub.add_parser("swap", help="Swap cached photo candidate rank and re-extract Gemma3 traits")
    p_swap.add_argument("--batch", required=True)
    p_swap.add_argument("stem")
    p_swap.add_argument("cand_rank", type=int)
    p_swap.set_defaults(func=cmd_swap)

    p_promote = sub.add_parser("promote", help="Promote approved eq_out PNGs to art-pipeline/masters/")
    p_promote.add_argument("--batch", required=True)
    p_promote.add_argument("--stems", nargs="*", default=[])
    p_promote.add_argument("--export-web", action="store_true")
    p_promote.set_defaults(func=cmd_promote)

    p_web = sub.add_parser("export-web", help="Export masters/*.png to 1024px frontend WebP icons")
    p_web.add_argument("--icons", default=str(MASTERS_DIR))
    p_web.add_argument("--out", default=str(WEB_ICONS_DIR))
    p_web.add_argument("--size", type=int, default=1024)
    p_web.add_argument("--quality", type=int, default=90)
    p_web.set_defaults(func=cmd_export_web)

    p_norm = sub.add_parser("normalize", help="Normalize transparent RGBA PNGs to 76%% fill square canvas")
    p_norm.add_argument("--in", dest="src", required=True)
    p_norm.add_argument("--out", required=True)
    p_norm.add_argument("--canvas", type=int, default=2048)
    p_norm.add_argument("--fill", type=float, default=0.76)
    p_norm.set_defaults(func=cmd_normalize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
