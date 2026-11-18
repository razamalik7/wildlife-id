"""End-to-end batch orchestrator for animl_art v7 with slicing, caching, QA flags & VRAM isolation."""

import gc
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from animl_art import (
    ART_ROOT,
    BATCHES_DIR,
    EXPANSION_LIST_PATH,
    LORA_OUTPUT_DIR,
    MASTERS_DIR,
    REPO_ROOT,
    RUNS_DIR,
    SDXL_BASE_PATH,
    WEB_ICONS_DIR,
)
from animl_art.matte import normalize_repaired_cutout
from animl_art.postprocess import (
    analyze_rebound_outline,
    clean_post_v7,
    equalize_outline_smooth_tapered,
    export_web_icons,
    load_font,
    measure_aa_midtone_pct,
    measure_shells_and_width,
    measure_trapped_fur_gap,
    render_comparison_boards,
)
from animl_art.prefetch import _slugify, infer_subtype
from animl_art.sources import fetch_wiki_clauses_map
from animl_art.stylize import (
    build_canny_from_repaired,
    compute_depth_maps,
    is_batoidea_ray_v7,
    run_sdxl_stylization,
)
from animl_art.vetter import (
    gather_top_candidates_for_species,
    unload_ollama_model,
    vet_side_by_side_and_extract_traits_v7,
)

SpeciesTuple = tuple[str, str, str, int, str, str, bool]

_ELASMOBRANCH_FALLBACKS: list[SpeciesTuple] = [
    ("10_nurse_shark", "Nurse Shark", "Ginglymostoma cirratum", 49964, "Elasmobranchii", "shark_ray", False),
    ("10_common_blacktip_shark", "Common Blacktip Shark", "Carcharhinus limbatus", 55529, "Elasmobranchii", "shark_ray", False),
]


def _update_error_log(work_dir: Path, stem: str, common: str, phase: str, error_msg: str | None) -> None:
    """Record or clear per-species fault in `work_dir / 'errors.json'`."""
    err_p = work_dir / "errors.json"
    rows = json.loads(err_p.read_text(encoding="utf-8")) if err_p.exists() else []
    rows = [r for r in rows if not (r.get("stem") == stem and r.get("phase") == phase)]
    if error_msg:
        rows.append({"stem": stem, "common": common, "phase": phase, "error": error_msg})
    err_p.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_species_specs(
    work_dir: Path,
    manifest_path: Path | None = None,
    species_queries: list[str] | None = None,
    sample_unseen: int = 0,
    offset: int = 0,
    limit: int = 0,
) -> list[SpeciesTuple]:
    """Resolve species list from cache, manifest JSON, CLI queries, or auto-sampling, applying `[offset:offset+limit]`."""
    specs: list[SpeciesTuple] = []
    resolved_p = work_dir / "manifest_resolved.json"
    if resolved_p.exists() and not manifest_path and not species_queries and not sample_unseen:
        specs = [tuple(r) for r in json.loads(resolved_p.read_text(encoding="utf-8"))]  # type: ignore[misc]
    elif manifest_path is not None:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for idx, item in enumerate(raw, start=1):
            if isinstance(item, list):
                specs.append(tuple(item))  # type: ignore[arg-type]
            else:
                com = item["common_name"]
                cls_n = item["class"]
                sub, d_perch = infer_subtype(com, cls_n)
                specs.append((
                    item.get("stem") or f"{idx:03d}_{_slugify(com)}",
                    com,
                    item["scientific_name"],
                    int(item["taxon_id"]),
                    cls_n,
                    item.get("subtype", sub),
                    bool(item.get("default_perched", d_perch)),
                ))
    elif species_queries:
        catalog = json.loads(EXPANSION_LIST_PATH.read_text(encoding="utf-8"))["species"]
        by_key: dict[str, dict] = {}
        for e in catalog:
            by_key[e["common_name"].lower()] = e
            by_key[e["scientific_name"].lower()] = e
            by_key[e["slug"].lower()] = e
        for idx, q in enumerate(species_queries, start=1):
            ql = q.strip().lower()
            e = by_key.get(ql) or by_key.get(ql.replace("-", "_"))
            if e is None:
                raise KeyError(f"Species {q!r} not found in {EXPANSION_LIST_PATH}")
            sub, d_perch = infer_subtype(e["common_name"], e["class"])
            specs.append((f"{idx:02d}_{_slugify(e['common_name'])}", e["common_name"], e["scientific_name"], int(e["taxon_id"]), e["class"], sub, d_perch))
    elif sample_unseen > 0:
        specs = sample_unseen_species(sample_unseen)
    elif (work_dir / "manifest.json").exists():
        specs = [tuple(r) for r in json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))]  # type: ignore[misc]
    else:
        raise ValueError("Provide --manifest, --species, or --sample-unseen (or an existing run manifest.json).")

    if offset > 0 or limit > 0:
        end = (offset + limit) if limit > 0 else None
        specs = specs[offset:end]
    return specs


def collect_used_species_keys() -> set[str]:
    """Gather all common names, scientific names, and stems already in masters, api config, batches, or archive."""
    used: set[str] = set()
    api_cfg = REPO_ROOT / "api" / "species_config.json"
    if api_cfg.exists():
        for entry in json.loads(api_cfg.read_text(encoding="utf-8")):
            for k in ("name", "common_name", "scientific_name"):
                if isinstance(entry.get(k), str):
                    used.add(entry[k].lower())
    for p in MASTERS_DIR.glob("*.png"):
        used.add(p.stem.lower().replace("_", " "))
    for bjson in list(BATCHES_DIR.glob("*.json")) + list(RUNS_DIR.glob("*/manifest*.json")):
        if "top500" in bjson.name:
            continue
        try:
            for item in json.loads(bjson.read_text(encoding="utf-8")):
                if isinstance(item, list) and len(item) >= 3:
                    used.update({str(item[1]).lower(), str(item[2]).lower()})
                elif isinstance(item, dict):
                    for k in ("common_name", "scientific_name"):
                        if isinstance(item.get(k), str):
                            used.add(item[k].lower())
        except Exception:
            pass
    for py_dir in (ART_ROOT / "archive" / "experiments",):
        if py_dir.exists():
            for pf in py_dir.glob("*.py"):
                txt = pf.read_text(encoding="utf-8", errors="ignore")
                for m in re.findall(r'"([A-Z][a-z]+ [a-z]{3,})"', txt):
                    used.add(m.lower())
    return used


def sample_unseen_species(count: int = 10) -> list[SpeciesTuple]:
    """Pick a balanced, strictly unseen batch across Aves, Mammalia, Reptilia, Amphibia, Elasmobranchii."""
    used = collect_used_species_keys()
    catalog = json.loads(EXPANSION_LIST_PATH.read_text(encoding="utf-8"))["species"]
    by_cls: dict[str, list[dict]] = {}
    for e in catalog:
        if e.get("include_reason") in ("current_100", "v4_style_anchor", "test_batch_20"):
            continue
        if e.get("id_mode") == "genus_candidates":
            continue
        if e["common_name"].lower() in used or e["scientific_name"].lower() in used:
            continue
        by_cls.setdefault(e["class"], []).append(e)
    for arr in by_cls.values():
        arr.sort(key=lambda x: -int(x.get("us_research_grade_obs", 0)))

    class_order = ["Mammalia", "Aves", "Reptilia", "Amphibia", "Elasmobranchii"]
    cursors = {c: 0 for c in class_order}
    picked: list[dict] = []
    while len(picked) < count:
        progressed = False
        for c in class_order:
            if len(picked) >= count:
                break
            arr = by_cls.get(c, [])
            if cursors[c] < len(arr):
                picked.append(arr[cursors[c]])
                cursors[c] += 1
                progressed = True
        if not progressed:
            break

    out: list[SpeciesTuple] = []
    for idx, e in enumerate(picked, start=1):
        com, sci, tid, cls_n = e["common_name"], e["scientific_name"], int(e["taxon_id"]), e["class"]
        sub, d_perch = infer_subtype(com, cls_n)
        out.append((f"{idx:02d}_{_slugify(com)}", com, sci, tid, cls_n, sub, d_perch))
    return out


def _merge_manifest_json(path: Path, new_specs: list[SpeciesTuple]) -> None:
    existing: dict[str, list] = {}
    if path.exists():
        for r in json.loads(path.read_text(encoding="utf-8")):
            existing[str(r[0])] = r
    for s in new_specs:
        existing[str(s[0])] = list(s)
    ordered = [existing[k] for k in sorted(existing.keys())]
    path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def run_phase_1a(work_dir: Path, species: list[SpeciesTuple], force_stems: set[str]) -> None:
    """Phase 1A: OpenCLIP + BiRefNet candidate cutout collection & caching with fault isolation."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    work_dir.mkdir(parents=True, exist_ok=True)
    _merge_manifest_json(work_dir / "manifest.json", species)
    cand_cache_dir = work_dir / "cands_top3"
    cand_cache_dir.mkdir(parents=True, exist_ok=True)

    for stem in sorted(force_stems):
        print(f"[Phase 1A force] Clearing cached artifacts for {stem}...", flush=True)
        shutil.rmtree(cand_cache_dir / stem, ignore_errors=True)
        for sub_dir in ("cutouts_1024", "cutouts_final", "raw_stylized", "canny_final", "depth_final", "clean_out", "eq_out"):
            (work_dir / sub_dir / f"{stem}.png").unlink(missing_ok=True)
        (work_dir / "metrics_by_stem" / f"{stem}.json").unlink(missing_ok=True)

    unload_ollama_model("gemma3:27b")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    resolved_manifest: list[SpeciesTuple] = []
    for idx, spec in enumerate(species):
        stem, common, _, _, _, subtype, _ = spec
        print(f"[Phase 1A {idx + 1}/{len(species)}] {stem} ({common})...", flush=True)
        try:
            saved = gather_top_candidates_for_species(spec, clip_model, clip_proc, device, cand_cache_dir)
            if len(saved) < 2 and subtype == "shark_ray":
                for fb in _ELASMOBRANCH_FALLBACKS:
                    fb_spec = (stem, fb[1], fb[2], fb[3], fb[4], fb[5], fb[6])
                    fb_saved = gather_top_candidates_for_species(fb_spec, clip_model, clip_proc, device, cand_cache_dir)
                    if len(fb_saved) >= 2:
                        spec, saved = fb_spec, fb_saved
                        break
            if not saved:
                raise RuntimeError("No valid cutouts found after fallback")
            resolved_manifest.append(spec)
            _update_error_log(work_dir, stem, common, "1a", None)
        except Exception as exc:
            print(f"  [Phase 1A ERROR] {stem} ({common}): {exc}", flush=True)
            _update_error_log(work_dir, stem, common, "1a", str(exc))

    _merge_manifest_json(work_dir / "manifest_resolved.json", resolved_manifest)
    print(f"Phase 1A complete for {work_dir.name}: {len(resolved_manifest)}/{len(species)} succeeded.", flush=True)


def run_phase_1b(
    work_dir: Path,
    species: list[SpeciesTuple],
    force_stems: set[str],
    boards_dir: Path | None = None,
) -> None:
    """Phase 1B: Wikipedia visual clauses + Gemma3:27b vetting & paginated Stage-1 SBS sheets."""
    import torch

    photo_dir = work_dir / "photos_selected"
    cutout_dir = work_dir / "cutouts_1024"
    sbs_dir = work_dir / "sbs_selected"
    cand_cache_dir = work_dir / "cands_top3"
    for d in (photo_dir, cutout_dir, sbs_dir):
        d.mkdir(parents=True, exist_ok=True)

    meta_path = work_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print(f"Resolving Wikipedia visual clauses for {len(species)} species (cache-first)...", flush=True)
    wiki_map = fetch_wiki_clauses_map([(s[1], s[2]) for s in species])

    for idx, (stem, common, sci, taxon_id, cls_name, subtype, _) in enumerate(species):
        out_cut = cutout_dir / f"{stem}.png"
        out_photo = photo_dir / f"{stem}.png"
        if stem in meta and stem not in force_stems and out_cut.exists() and out_photo.exists():
            print(f"[Phase 1B {idx + 1}/{len(species)}] {stem} already vetted.", flush=True)
            continue

        cands_p = cand_cache_dir / stem / "cands.json"
        if not cands_p.exists():
            _update_error_log(work_dir, stem, common, "1b", "Missing Phase 1A candidates")
            continue

        try:
            saved_cands = json.loads(cands_p.read_text(encoding="utf-8"))
            best_choice = None
            best_total = -999.0
            wiki_c = wiki_map.get(common, "")
            print(f"[Phase 1B {idx + 1}/{len(species)}] Gemma3:27b vetting {common} ({sci})...", flush=True)

            for c_info in saved_cands:
                rank_i = c_info["rank"]
                norm_cut = Image.open(cand_cache_dir / stem / f"cut_{rank_i}.png").convert("RGB")
                pad_photo = Image.open(cand_cache_dir / stem / f"photo_{rank_i}.png").convert("RGB")
                clip_s = float(c_info["clip_score"])
                vlm_res = vet_side_by_side_and_extract_traits_v7(pad_photo, norm_cut, common, sci, subtype, wiki_c)
                valid_live = bool(vlm_res.get("valid_live_wild_pose", False))
                no_blob = bool(vlm_res.get("no_fused_background_blob", False))
                no_glass = bool(vlm_res.get("no_aquarium_glass_reflection", True))
                pose_s = float(vlm_res.get("pose_quality_score", 0.0))
                total_s = (
                    pose_s
                    + clip_s * 2.5
                    - (0.0 if valid_live else 8.0)
                    - (0.0 if no_blob else 5.0)
                    - (0.0 if (no_glass or subtype != "shark_ray") else 5.0)
                )
                print(
                    f"  cand #{rank_i} id={c_info['photo_id']}: live={valid_live} no_blob={no_blob} "
                    f"no_glass={no_glass} pose={pose_s:.1f} repair={vlm_res.get('cutout_repair_needed')} -> {total_s:.2f}",
                    flush=True,
                )
                if total_s > best_total:
                    best_total = total_s
                    best_choice = (norm_cut, pad_photo, c_info, vlm_res)
                if valid_live and no_blob and (no_glass or subtype != "shark_ray") and pose_s >= 7.5:
                    break

            norm_cut, pad_photo, c_info, vlm_res = best_choice  # type: ignore[misc]
            pad_photo.save(out_photo)
            norm_cut.save(out_cut)
            sbs = Image.new("RGB", (1024, 512), (255, 255, 255))
            sbs.paste(pad_photo.resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
            sbs.paste(norm_cut.resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
            sbs.save(sbs_dir / f"{stem}.jpg", quality=88)

            repair_need = vlm_res.get("cutout_repair_needed", "none")
            meta[stem] = {
                "common_name": common,
                "scientific_name": sci,
                "taxon_id": taxon_id,
                "class": cls_name,
                "subtype": subtype,
                "photo_id": c_info["photo_id"],
                "photo_url": c_info["original_url"],
                "pose_quality_score": vlm_res.get("pose_quality_score"),
                "cutout_repair_needed": repair_need,
                "has_perch_stub": (subtype == "bird_perched" and repair_need != "perch_stub_under_feet"),
                "eye_color_in_photo": vlm_res.get("eye_color_in_photo", ""),
                "visual_traits": vlm_res.get("visual_traits", ""),
                "wiki_clauses": wiki_c,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            _update_error_log(work_dir, stem, common, "1b", None)
        except Exception as exc:
            print(f"  [Phase 1B ERROR] {stem} ({common}): {exc}", flush=True)
            _update_error_log(work_dir, stem, common, "1b", str(exc))

    unload_ollama_model("gemma3:27b")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sbs_out = boards_dir or (work_dir / "boards")
    sbs_out.mkdir(parents=True, exist_ok=True)
    font_lbl = load_font(14, bold=True)
    for page_idx, start in enumerate(range(0, len(species), 20), start=1):
        sub_sp = species[start : start + 20]
        n_rows = max(1, (len(sub_sp) + 1) // 2)
        sheet = Image.new("RGB", (1600, n_rows * 210), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, (stem, common, _, _, _, subtype, _) in enumerate(sub_sp):
            r, col = divmod(i, 2)
            x0, y0 = col * 800, r * 210
            sbs_p = sbs_dir / f"{stem}.jpg"
            if sbs_p.exists():
                sheet.paste(Image.open(sbs_p).resize((780, 178), Image.Resampling.LANCZOS), (x0 + 10, y0 + 26))
            rep = meta.get(stem, {}).get("cutout_repair_needed", "none")
            draw.rectangle([x0, y0, x0 + 800, y0 + 210], outline=(220, 224, 230), width=1)
            draw.text((x0 + 12, y0 + 5), f"{stem} ({common}) [{subtype}] repair={rep}", fill=(20, 24, 30), font=font_lbl)
        suffix = f"_p{page_idx:02d}" if len(species) > 20 else ""
        sheet_p = sbs_out / f"v7_{work_dir.name}_stage1_sbs{suffix}.jpg"
        sheet.save(sheet_p, quality=92)
        print(f"Saved Stage 1 SBS sheet: {sheet_p}", flush=True)


def _compute_qa_flags(row: dict, meta_entry: dict) -> list[str]:
    flags: list[str] = []
    if float(row.get("black_line_pct", 100.0)) < 93.0:
        flags.append("low_black_outline")
    if float(row.get("trapped_fur_gap_pct", 0.0)) > 2.2:
        flags.append("fur_halo_gap")
    if float(row.get("aa_midtone_pct", 100.0)) < 70.0:
        flags.append("low_aa")
    pose_s = meta_entry.get("pose_quality_score")
    if pose_s is not None and float(pose_s) < 6.5:
        flags.append("low_pose_score")
    rep = meta_entry.get("cutout_repair_needed", "none")
    if rep and rep != "none":
        flags.append(f"repair:{rep}")
    return flags


def run_phase_stylize(
    work_dir: Path,
    species: list[SpeciesTuple],
    force_stems: set[str],
    unet_only_stems: set[str],
    boards_dir: Path | None = None,
) -> None:
    """Phase 3: Depth + Canny + SDXL v7 + incremental post-clean, v3 outline equalizer & QA summary."""
    import torch

    meta_path = work_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    unload_ollama_model("gemma3:27b")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cut_final_dir = work_dir / "cutouts_final"
    canny_dir = work_dir / "canny_final"
    depth_dir = work_dir / "depth_final"
    repaired_dir = work_dir / "repaired_cutouts"
    metrics_dir = work_dir / "metrics_by_stem"
    for d in (cut_final_dir, canny_dir, depth_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    changed_depth: list[str] = []
    ready_species: list[SpeciesTuple] = []
    for spec in species:
        stem, common, _, _, _, subtype, _ = spec
        if stem not in meta:
            continue
        rep_p = repaired_dir / f"{stem}.png"
        src_cut = work_dir / "cutouts_1024" / f"{stem}.png"
        if not rep_p.exists() and not src_cut.exists():
            continue

        cut_final_p = cut_final_dir / f"{stem}.png"
        canny_p = canny_dir / f"{stem}.png"
        depth_p = depth_dir / f"{stem}.png"
        if rep_p.exists():
            cut_im = normalize_repaired_cutout(Image.open(rep_p))
            cut_im.save(cut_final_p)
            if subtype == "bird_perched":
                meta[stem]["has_perch_stub"] = True
            changed_depth.append(stem)
        elif not cut_final_p.exists():
            cut_im = Image.open(src_cut).convert("RGB")
            cut_im.save(cut_final_p)
            changed_depth.append(stem)
        else:
            cut_im = Image.open(cut_final_p).convert("RGB")
            if not depth_p.exists():
                changed_depth.append(stem)

        if not canny_p.exists() or rep_p.exists():
            build_canny_from_repaired(cut_im).save(canny_p)
        ready_species.append(spec)

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    compute_depth_maps(changed_depth, cut_final_dir, depth_dir)

    items = [
        {
            "stem": stem,
            "common": common,
            "sci": sci,
            "cls_name": cls_name,
            "subtype": subtype,
            "is_perched": bool(meta.get(stem, {}).get("has_perch_stub", False)),
            "photo_p": work_dir / "photos_selected" / f"{stem}.png",
            "cut_p": cut_final_dir / f"{stem}.png",
            "canny_p": canny_dir / f"{stem}.png",
            "depth_p": depth_dir / f"{stem}.png",
        }
        for stem, common, sci, _, cls_name, subtype, _ in ready_species
    ]

    raw_dir = work_dir / "raw_stylized"
    clean_dir = work_dir / "clean_out"
    eq_dir = work_dir / "eq_out"
    for d in (raw_dir, clean_dir, eq_dir):
        d.mkdir(parents=True, exist_ok=True)

    need_render = [it for it in items if (not (raw_dir / f"{it['stem']}.png").exists()) or (it["stem"] in force_stems)]
    run_sdxl_stylization(need_render, meta, raw_dir, LORA_OUTPUT_DIR, SDXL_BASE_PATH, unet_only_stems)

    print(f"\n=== Post-Clean (`clean_post_v7`) + v3 Outline Equalizer ({work_dir.name}) ===", flush=True)
    for it in items:
        stem = it["stem"]
        eq_p = eq_dir / f"{stem}.png"
        m_p = metrics_dir / f"{stem}.json"
        if eq_p.exists() and m_p.exists() and stem not in force_stems:
            continue
        raw_p = raw_dir / f"{stem}.png"
        if not raw_p.exists():
            continue
        try:
            raw_im = Image.open(raw_p).convert("RGB")
            cut_im = Image.open(it["cut_p"]).convert("RGB")
            clean_p = clean_dir / f"{stem}.png"

            clean_post_v7(
                raw_im,
                cut_im,
                clean_p,
                is_perched=it["is_perched"],
                preserve_thin_appendages=(it["subtype"] in ("shark_ray", "mammal_aquatic")),
            )
            eq_rgb, eq_stats = equalize_outline_smooth_tapered(clean_p, strength=0.90)
            eq_rgb[:20, :] = eq_rgb[-20:, :] = eq_rgb[:, :20] = eq_rgb[:, -20:] = 255
            Image.fromarray(eq_rgb).save(eq_p)

            aa_pct = measure_aa_midtone_pct(eq_rgb)
            gap_pct = measure_trapped_fur_gap(eq_rgb)
            reb = analyze_rebound_outline(eq_p)
            sh = measure_shells_and_width(eq_p)
            row = {
                "stem": stem,
                "common": it["common"],
                "class": it["cls_name"],
                "is_perched": it["is_perched"],
                "unet_only": (stem in unet_only_stems) or is_batoidea_ray_v7(it["common"], it["cls_name"]),
                "aa_midtone_pct": round(aa_pct, 2),
                "trapped_fur_gap_pct": round(gap_pct, 2),
                "black_line_pct": round(reb["black_pct"], 2),
                "total_line_pct": round(reb["total_pct"], 2),
                "width_px": round(sh["width_px"], 2),
                "rim3_l": round(sh["rim3_l"], 2),
                "int_crease_px": int(eq_stats["int_crease_px"]),
            }
            row["qa_flags"] = _compute_qa_flags(row, meta.get(stem, {}))
            m_p.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(
                f"  {stem:28s} | AA={aa_pct:5.1f}% Gap={gap_pct:4.1f}% "
                f"Blk={reb['black_pct']:5.1f}% Tot={reb['total_pct']:5.1f}% W={sh['width_px']:4.2f} "
                f"{'FLAG:' + ','.join(row['qa_flags']) if row['qa_flags'] else 'OK'}",
                flush=True,
            )
            _update_error_log(work_dir, stem, it["common"], "stylize", None)
        except Exception as exc:
            print(f"  [Postprocess ERROR] {stem} ({it['common']}): {exc}", flush=True)
            _update_error_log(work_dir, stem, it["common"], "stylize", str(exc))

    all_rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(metrics_dir.glob("*.json"))]
    if all_rows:
        summary = {
            "count": len(all_rows),
            "mean_aa_pct": round(float(np.mean([r["aa_midtone_pct"] for r in all_rows])), 2),
            "mean_gap_pct": round(float(np.mean([r["trapped_fur_gap_pct"] for r in all_rows])), 2),
            "mean_black_pct": round(float(np.mean([r["black_line_pct"] for r in all_rows])), 2),
            "mean_total_pct": round(float(np.mean([r["total_line_pct"] for r in all_rows])), 2),
            "mean_width_px": round(float(np.mean([r["width_px"] for r in all_rows])), 2),
            "mean_int_px": round(float(np.mean([r["int_crease_px"] for r in all_rows])), 1),
        }
        (work_dir / "metrics.json").write_text(json.dumps({"summary": summary, "rows": all_rows}, indent=2), encoding="utf-8")
        flagged = [{"stem": r["stem"], "common": r["common"], "flags": r.get("qa_flags", [])} for r in all_rows if r.get("qa_flags")]
        qa_sum = {
            "total": len(all_rows),
            "clean_count": len(all_rows) - len(flagged),
            "flagged_count": len(flagged),
            "flagged": flagged,
        }
        (work_dir / "qa_summary.json").write_text(json.dumps(qa_sum, indent=2), encoding="utf-8")

    current_stems = {it["stem"] for it in items}
    cur_rows = [r for r in all_rows if r["stem"] in current_stems]
    if cur_rows:
        render_comparison_boards(work_dir.name, items, cur_rows, raw_dir, eq_dir, boards_dir or (work_dir / "boards"))


def swap_candidate(work_dir: Path, stem: str, cand_rank: int) -> None:
    """Swap cached candidate `cand_rank` into active cutout/photo and re-run Gemma3:27b trait extraction."""
    species = load_species_specs(work_dir)
    spec = next(s for s in species if s[0] == stem)
    _, common, sci, _, _, subtype, _ = spec
    cand_dir = work_dir / "cands_top3" / stem
    cands = json.loads((cand_dir / "cands.json").read_text(encoding="utf-8"))
    c_info = next(c for c in cands if c["rank"] == cand_rank)

    norm_cut = Image.open(cand_dir / f"cut_{cand_rank}.png").convert("RGB")
    pad_photo = Image.open(cand_dir / f"photo_{cand_rank}.png").convert("RGB")
    (work_dir / "photos_selected").mkdir(parents=True, exist_ok=True)
    (work_dir / "cutouts_1024").mkdir(parents=True, exist_ok=True)
    (work_dir / "sbs_selected").mkdir(parents=True, exist_ok=True)
    pad_photo.save(work_dir / "photos_selected" / f"{stem}.png")
    norm_cut.save(work_dir / "cutouts_1024" / f"{stem}.png")
    for stale_dir in ("cutouts_final", "canny_final", "depth_final", "raw_stylized", "clean_out", "eq_out"):
        (work_dir / stale_dir / f"{stem}.png").unlink(missing_ok=True)
    (work_dir / "metrics_by_stem" / f"{stem}.json").unlink(missing_ok=True)

    sbs = Image.new("RGB", (1024, 512), (255, 255, 255))
    sbs.paste(pad_photo.resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
    sbs.paste(norm_cut.resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
    sbs.save(work_dir / "sbs_selected" / f"{stem}.jpg", quality=88)

    meta_path = work_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wiki_c = meta.get(stem, {}).get("wiki_clauses") or fetch_wiki_clauses_map([(common, sci)]).get(common, "")
    res = vet_side_by_side_and_extract_traits_v7(pad_photo, norm_cut, common, sci, subtype, wiki_c)
    repair_need = res.get("cutout_repair_needed", "none")
    meta[stem].update(
        {
            "photo_id": c_info["photo_id"],
            "photo_url": c_info["original_url"],
            "pose_quality_score": res.get("pose_quality_score"),
            "cutout_repair_needed": repair_need,
            "has_perch_stub": (subtype == "bird_perched" and repair_need != "perch_stub_under_feet"),
            "eye_color_in_photo": res.get("eye_color_in_photo", ""),
            "visual_traits": res.get("visual_traits", ""),
            "wiki_clauses": wiki_c,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    unload_ollama_model("gemma3:27b")
    print(f"Swapped {stem} -> cand #{cand_rank} ({c_info['photo_id']})", flush=True)


def promote_to_masters(work_dir: Path, stems: list[str] | None = None, export_web: bool = False) -> int:
    """Promote final equalized PNGs from `work_dir/eq_out` into `art-pipeline/masters/`."""
    eq_dir = work_dir / "eq_out"
    meta_path = work_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    targets = sorted(eq_dir.glob("*.png"))
    if stems:
        wanted = set(stems)
        targets = [p for p in targets if p.stem in wanted]
    MASTERS_DIR.mkdir(parents=True, exist_ok=True)
    promoted = 0
    for p in targets:
        common = meta.get(p.stem, {}).get("common_name") or re.sub(r"^\d+_", "", p.stem)
        slug = _slugify(common)
        dst = MASTERS_DIR / f"{slug}.png"
        shutil.copy2(p, dst)
        promoted += 1
        print(f"  Promoted {p.name} -> {dst.relative_to(REPO_ROOT)}")
    if export_web and promoted:
        export_web_icons(MASTERS_DIR, WEB_ICONS_DIR)
    return promoted


def apply_review_decisions(work_dir: Path, promote: bool = False, export_web: bool = False) -> dict:
    """Apply queued swaps, reruns, and approvals recorded in `work_dir / 'review_decisions.json'`."""
    dec_path = work_dir / "review_decisions.json"
    if not dec_path.exists():
        return {"swapped": 0, "rerun": 0, "promoted": 0}
    decisions: dict = json.loads(dec_path.read_text(encoding="utf-8"))

    force_stems: set[str] = set()
    unet_stems: set[str] = set()
    swapped_cnt = 0
    for stem, dec in sorted(decisions.items()):
        act = dec.get("action")
        if dec.get("unet_only"):
            unet_stems.add(stem)
        if act == "swap" and dec.get("swap_rank") is not None:
            swap_candidate(work_dir, stem, int(dec["swap_rank"]))
            force_stems.add(stem)
            swapped_cnt += 1
        elif act == "rerun":
            force_stems.add(stem)

    if force_stems:
        all_specs = load_species_specs(work_dir)
        sub_specs = [s for s in all_specs if s[0] in force_stems]
        run_phase_stylize(work_dir, sub_specs, force_stems=force_stems, unet_only_stems=unet_stems)

    promoted_cnt = 0
    if promote:
        approved = [stem for stem, dec in decisions.items() if dec.get("action") == "approve" and (work_dir / "eq_out" / f"{stem}.png").exists()]
        if approved:
            promoted_cnt = promote_to_masters(work_dir, stems=approved, export_web=export_web)
    return {"swapped": swapped_cnt, "rerun": len(force_stems), "promoted": promoted_cnt}


def run_phase_all_isolated(
    cli_path: Path,
    batch_name: str,
    manifest_path: Path | None,
    species_queries: list[str] | None,
    sample_unseen: int,
    offset: int,
    limit: int,
    force_stems: list[str],
    boards_dir: Path | None,
) -> None:
    """Run phases `1a`, `1b`, and `stylize` in fresh subprocesses so VRAM clears between stages."""
    base_cmd = [sys.executable, str(cli_path), "run", "--batch", batch_name]
    if manifest_path:
        base_cmd += ["--manifest", str(manifest_path)]
    if species_queries:
        base_cmd += ["--species", *species_queries]
    if sample_unseen > 0:
        base_cmd += ["--sample-unseen", str(sample_unseen)]
    if offset > 0:
        base_cmd += ["--offset", str(offset)]
    if limit > 0:
        base_cmd += ["--limit", str(limit)]
    if force_stems:
        base_cmd += ["--force", *force_stems]
    if boards_dir:
        base_cmd += ["--boards-out", str(boards_dir)]

    for phase in ("1a", "1b", "stylize"):
        cmd = base_cmd + ["--phase", phase]
        print(f"\n==================== ISOLATED SUBPROCESS: PHASE {phase.upper()} ====================", flush=True)
        subprocess.run(cmd, check=True)
