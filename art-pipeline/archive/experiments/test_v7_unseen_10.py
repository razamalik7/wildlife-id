"""End-to-End Automated Evaluation of `animl_style_v7 (step 2000)` on 10 Strictly Unseen Species.

Zero overlap with `v6` (25 masters), `v7` (45 images), `current_100` masters, `Batch 1`, `Batch 2`, or `eq78`.

VRAM-safe 3-phase execution on 16GB RTX 5080:
  Phase 1A (`--phase 1a`): OpenCLIP ViT-B-32 + BiRefNet candidate cutout collection -> exits process.
  Phase 1B (`--phase 1b`): Wikipedia visual clauses + Gemma3:27b side-by-side vetting & trait extraction -> unloads Ollama.
  Stage 2 (manual/subagent check): Inspect `v7_unseen10_stage1_sbs.jpg`; place any Stage 2 repaired cutouts in `ARTIFACT_DIR/genai_u10_{stem}.png`.
  Phase 3  (`--phase stylize`): Depth-Anything-V2 + Canny + SDXL + MultiControlNet + `animl_style_v7-step00002000.safetensors`
                                -> unload SDXL -> `clean_post_v7` + `v3` Smooth Outline Equalizer -> 2 comparison boards.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from measure_outline_trough import analyze_rebound_outline
from pipeline_v7 import (
    build_prompt_and_neg_v7,
    clean_post_v7,
    fetch_marine_candidates_v7,
    fetch_wiki_clauses_map,
    is_batoidea_ray_v7,
    make_normalized_cutout_v7,
    prepare_conditioned_cutout_and_canny_v7,
    vet_side_by_side_and_extract_traits_v7,
)
from smooth_outline_equalizer import (
    equalize_outline_smooth_tapered,
    measure_aa_midtone_pct,
    measure_trapped_fur_gap,
)
from test_fresh_20_batch2 import download_pil_compliant, fetch_commons_files, normalize_repaired_cutout
from test_fresh_20_e2e import (
    encode_prompt_sdxl_multichunk,
    fetch_inat_curated_taxon,
    fetch_inat_obs,
    make_normalized_cutout,
    strip_head_eye_clauses,
    unload_ollama_model,
)
from test_pre_stylization_crop_repair import build_canny_from_repaired
from verify_width_and_shells import measure_shells_and_width

ROOT = Path(__file__).parent
LORA = ROOT / "lora"
WORK = ROOT / "batch_run_v7_unseen_10"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

# 10 Strictly Unseen North American Species (stem, common, sci, taxon_id, cls_name, subtype, default_perched)
UNSEEN_10_SPECIES: list[tuple[str, str, str, int, str, str, bool]] = [
    ("01_long_tailed_weasel", "Long-tailed Weasel", "Neogale frenata", 1264431, "Mammalia", "mammal_ground", False),
    ("02_hoary_marmot", "Hoary Marmot", "Marmota caligata", 46087, "Mammalia", "mammal_ground", False),
    ("03_steller_sea_lion", "Steller Sea Lion", "Eumetopias jubatus", 41755, "Mammalia", "mammal_ground", False),
    ("04_eastern_bluebird", "Eastern Bluebird", "Sialia sialis", 12942, "Aves", "bird_perched", True),
    ("05_killdeer", "Killdeer", "Charadrius vociferus", 4793, "Aves", "bird_ground", False),
    ("06_red_shouldered_hawk", "Red-shouldered Hawk", "Buteo lineatus", 5206, "Aves", "bird_perched", True),
    ("07_common_five_lined_skink", "Common Five-lined Skink", "Plestiodon fasciatus", 73788, "Reptilia", "lizard", False),
    ("08_western_ribbon_snake", "Western Ribbon Snake", "Thamnophis proximus", 60342, "Reptilia", "snake", False),
    ("09_marbled_salamander", "Marbled Salamander", "Ambystoma opacum", 26736, "Amphibia", "amphibian", False),
    ("10_spotted_eagle_ray", "Spotted Eagle Ray", "Aetobatus narinari", 49297, "Elasmobranchii", "shark_ray", False),
]

ELASMOBRANCH_FALLBACKS: list[tuple[str, str, str, int, str, str, bool]] = [
    ("10_nurse_shark", "Nurse Shark", "Ginglymostoma cirratum", 49964, "Elasmobranchii", "shark_ray", False),
    ("10_common_blacktip_shark", "Common Blacktip Shark", "Carcharhinus limbatus", 55529, "Elasmobranchii", "shark_ray", False),
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
        if bold
        else [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
    )
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def load_resolved_manifest() -> list[tuple[str, str, str, int, str, str, bool]]:
    manifest_p = WORK / "unseen10_manifest.json"
    if manifest_p.exists():
        rows = json.loads(manifest_p.read_text(encoding="utf-8"))
        return [tuple(r) for r in rows]  # type: ignore[misc]
    return list(UNSEEN_10_SPECIES)


def _gather_top3_for_species(
    spec: tuple[str, str, str, int, str, str, bool],
    clip_model,
    clip_proc,
    device: str,
    cand_cache_dir: Path,
) -> list[dict]:
    stem, common, sci, taxon_id, cls_name, subtype, _ = spec
    sp_dir = cand_cache_dir / stem
    sp_meta = sp_dir / "cands.json"
    if sp_meta.exists():
        return json.loads(sp_meta.read_text(encoding="utf-8"))

    print(f"  Fetching & segmenting candidates for {common} ({sci})...", flush=True)
    if subtype == "shark_ray":
        cands = fetch_marine_candidates_v7(taxon_id, sci, common)
        max_try, target_valid, keep_top = 24, 12, 4
    else:
        cands = []
        seen = set()
        for c in fetch_inat_curated_taxon(taxon_id, limit=8):
            if c["photo_id"] not in seen:
                seen.add(c["photo_id"])
                cands.append(c)
        for c in fetch_commons_files(sci, common, limit=6):
            if c["photo_id"] not in seen:
                seen.add(c["photo_id"])
                cands.append(c)
        try:
            for c in fetch_inat_obs(taxon_id, "votes", 12):
                if c["photo_id"] not in seen:
                    seen.add(c["photo_id"])
                    cands.append(c)
        except Exception:
            pass
        max_try, target_valid, keep_top = 18, 7, 3

    valid_packs = []
    for c in cands[:max_try]:
        im = download_pil_compliant(c["medium_url"], timeout=12) or download_pil_compliant(c["original_url"], timeout=15)
        if im is None or min(im.size) < 300:
            continue
        norm_cut, pad_photo, st = make_normalized_cutout_v7(im, subtype=subtype)
        if not st.get("valid") or norm_cut is None:
            continue
        valid_packs.append((norm_cut, pad_photo, c, st))
        if len(valid_packs) >= target_valid:
            break

    if not valid_packs:
        return []

    if subtype == "shark_ray":
        scuba_pos = (
            f"a clear underwater scuba diving photograph of a single live {common} ({sci}) "
            f"swimming submerged in blue ocean water showing its spotted back and long tail"
        )
        scuba_neg = (
            "photograph taken from above the water surface looking down through green water ripples and glare, "
            "upside-down white belly view in aquarium tunnel, beached dead fish on sand or boat"
        )
        cut_pos = (
            f"a complete full-body side-three-quarter view of a {common} ({sci}) "
            f"with both wings and long whip tail intact on a white background"
        )
        cut_neg = "cropped half body, missing tail, tailless blob, upside-down white belly"
        with torch.no_grad():
            in_photo = clip_proc(
                text=[scuba_pos, scuba_neg],
                images=[p[1] for p in valid_packs],
                return_tensors="pt",
                padding=True,
            ).to(device)
            scuba_prob = clip_model(**in_photo).logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()
            in_cut = clip_proc(
                text=[cut_pos, cut_neg],
                images=[p[0] for p in valid_packs],
                return_tensors="pt",
                padding=True,
            ).to(device)
            cut_prob = clip_model(**in_cut).logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()
        logits = 0.55 * scuba_prob + 0.45 * cut_prob
    else:
        pos_txt = f"a sharp full-body side-view wildlife field guide photo of a single {common} ({sci}) isolated on a white background"
        neg_txt = "cropped half body, human hand holding animal, dead animal, blurry, multiple animals, murky underwater"
        with torch.no_grad():
            inputs = clip_proc(
                text=[pos_txt, neg_txt],
                images=[p[0] for p in valid_packs],
                return_tensors="pt",
                padding=True,
            ).to(device)
            logits = clip_model(**inputs).logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()

    ranked_indices = np.argsort(logits)[::-1][:keep_top]
    sp_dir.mkdir(parents=True, exist_ok=True)
    saved_cands = []
    for rank_i, r_idx in enumerate(ranked_indices):
        norm_cut, pad_photo, c, _ = valid_packs[int(r_idx)]
        clip_s = float(logits[int(r_idx)])
        im_hi = download_pil_compliant(c["original_url"], timeout=16)
        if im_hi is not None and max(im_hi.size) > 600:
            hi_cut, hi_pad, hi_st = make_normalized_cutout_v7(im_hi, subtype=subtype)
            if hi_st.get("valid") and hi_cut is not None:
                norm_cut, pad_photo = hi_cut, hi_pad
        norm_cut.save(sp_dir / f"cut_{rank_i}.png")
        pad_photo.save(sp_dir / f"photo_{rank_i}.png")
        saved_cands.append(
            {
                "rank": rank_i,
                "photo_id": c["photo_id"],
                "original_url": c["original_url"],
                "clip_score": round(clip_s, 4),
            }
        )
    sp_meta.write_text(json.dumps(saved_cands, indent=2), encoding="utf-8")
    return saved_cands


def run_phase_1a(force_species: set[str] | None = None) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cand_cache_dir = WORK / "cands_top3"
    cand_cache_dir.mkdir(parents=True, exist_ok=True)

    for stem in sorted(force_species or set()):
        print(f"[Phase 1A force] Clearing cached artifacts for {stem}...", flush=True)
        shutil.rmtree(cand_cache_dir / stem, ignore_errors=True)
        for sub_dir in ("cutouts_1024", "cutouts_final", "raw_stylized", "canny_final", "depth_final", "clean_out", "eq_out"):
            (WORK / sub_dir / f"{stem}.png").unlink(missing_ok=True)
        genai_p = ARTIFACT_DIR / f"genai_u10_{stem}.png"
        if genai_p.exists():
            genai_p.unlink()

    unload_ollama_model("gemma3:27b")
    torch.cuda.empty_cache()

    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    resolved_manifest: list[tuple[str, str, str, int, str, str, bool]] = []
    for idx, spec in enumerate(UNSEEN_10_SPECIES):
        stem, common, _, _, _, _, _ = spec
        print(f"[Phase 1A {idx + 1}/10] {stem} ({common})...", flush=True)
        saved = _gather_top3_for_species(spec, clip_model, clip_proc, device, cand_cache_dir)
        if len(saved) < 2 and stem == "10_spotted_eagle_ray":
            print(f"  Only {len(saved)} clean cutouts for Spotted Eagle Ray; checking elasmobranch fallbacks...", flush=True)
            for fb in ELASMOBRANCH_FALLBACKS:
                fb_saved = _gather_top3_for_species(fb, clip_model, clip_proc, device, cand_cache_dir)
                if len(fb_saved) >= 2:
                    print(f"  Using fallback #{10}: {fb[1]} ({fb[2]}) with {len(fb_saved)} cutouts.", flush=True)
                    spec = fb
                    saved = fb_saved
                    break
        if not saved:
            raise RuntimeError(f"No valid cutouts found for {stem} ({common})")
        resolved_manifest.append(spec)

    (WORK / "unseen10_manifest.json").write_text(json.dumps(resolved_manifest, indent=2), encoding="utf-8")
    print("Phase 1A complete! Saved unseen10_manifest.json and top-3 cutouts. Exiting cleanly.", flush=True)


def run_phase_1b(force_revet_stems: set[str] | None = None) -> None:
    photo_dir = WORK / "photos_selected"
    cutout_dir = WORK / "cutouts_1024"
    sbs_dir = WORK / "sbs_selected"
    cand_cache_dir = WORK / "cands_top3"
    for d in (photo_dir, cutout_dir, sbs_dir):
        d.mkdir(parents=True, exist_ok=True)

    species = load_resolved_manifest()
    meta_path = WORK / "unseen10_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    revet = force_revet_stems or set()

    print("Fetching upgraded Wikipedia visual clauses for the 10 unseen species...", flush=True)
    wiki_map = fetch_wiki_clauses_map([(s[1], s[2]) for s in species])

    for idx, (stem, common, sci, taxon_id, cls_name, subtype, _) in enumerate(species):
        out_cut = cutout_dir / f"{stem}.png"
        out_photo = photo_dir / f"{stem}.png"
        if stem in meta and stem not in revet and out_cut.exists() and out_photo.exists():
            print(f"[Phase 1B {idx + 1}/10] {stem} already vetted (score={meta[stem].get('pose_quality_score')}).", flush=True)
            continue

        saved_cands = json.loads((cand_cache_dir / stem / "cands.json").read_text(encoding="utf-8"))
        best_choice = None
        best_total = -999.0
        wiki_c = wiki_map.get(common, "")
        print(f"[Phase 1B {idx + 1}/10] Gemma3:27b vetting {common} ({sci}) [wiki len={len(wiki_c)}]...", flush=True)

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
                best_choice = (norm_cut, pad_photo, c_info, vlm_res, total_s)
            if valid_live and no_blob and (no_glass or subtype != "shark_ray") and pose_s >= 7.5:
                break

        norm_cut, pad_photo, c_info, vlm_res, total_s = best_choice  # type: ignore[misc]
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
        print(f"  -> selected traits: {meta[stem]['visual_traits']!r}", flush=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    unload_ollama_model("gemma3:27b")
    gc.collect()
    torch.cuda.empty_cache()

    # Build 10-row Stage 1 SBS contact sheet for mandatory inspection before Stage 2 / Stylize
    sheet = Image.new("RGB", (1600, 1050), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font_lbl = load_font(14, bold=True)
    for i, (stem, common, sci, _, _, subtype, _) in enumerate(species):
        r, col = divmod(i, 2)
        x0, y0 = col * 800, r * 210
        sbs_p = sbs_dir / f"{stem}.jpg"
        if sbs_p.exists():
            sheet.paste(Image.open(sbs_p).resize((780, 178), Image.Resampling.LANCZOS), (x0 + 10, y0 + 26))
        rep = meta.get(stem, {}).get("cutout_repair_needed", "none")
        draw.rectangle([x0, y0, x0 + 800, y0 + 210], outline=(220, 224, 230), width=1)
        draw.text((x0 + 12, y0 + 5), f"{stem} ({common}) [{subtype}] repair={rep}", fill=(20, 24, 30), font=font_lbl)

    sheet_p = ARTIFACT_DIR / "v7_unseen10_stage1_sbs.jpg"
    sheet.save(sheet_p, quality=92)
    print(f"Saved Stage 1 SBS inspection sheet: {sheet_p}", flush=True)


def run_swap_candidate(stem: str, cand_rank: int) -> None:
    """Swap `cutouts_1024/{stem}.png` and `photos_selected/{stem}.png` to candidate `cand_rank` and re-extract traits."""
    species = load_resolved_manifest()
    spec = next(s for s in species if s[0] == stem)
    _, common, sci, _, _, subtype, _ = spec
    cand_dir = WORK / "cands_top3" / stem
    cands = json.loads((cand_dir / "cands.json").read_text(encoding="utf-8"))
    c_info = next(c for c in cands if c["rank"] == cand_rank)

    norm_cut = Image.open(cand_dir / f"cut_{cand_rank}.png").convert("RGB")
    pad_photo = Image.open(cand_dir / f"photo_{cand_rank}.png").convert("RGB")
    pad_photo.save(WORK / "photos_selected" / f"{stem}.png")
    norm_cut.save(WORK / "cutouts_1024" / f"{stem}.png")
    sbs = Image.new("RGB", (1024, 512), (255, 255, 255))
    sbs.paste(pad_photo.resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
    sbs.paste(norm_cut.resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
    sbs.save(WORK / "sbs_selected" / f"{stem}.jpg", quality=88)

    meta_path = WORK / "unseen10_metadata.json"
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
    print(f"Swapped {stem} -> cand #{cand_rank} ({c_info['photo_id']}): traits={meta[stem]['visual_traits']!r}", flush=True)


def prepare_final_cutouts_and_conditions(species: list[tuple[str, str, str, int, str, str, bool]], meta: dict) -> list[dict]:
    from transformers import pipeline as hf_pipeline

    cut_final_dir = WORK / "cutouts_final"
    canny_dir = WORK / "canny_final"
    depth_dir = WORK / "depth_final"
    for d in (cut_final_dir, canny_dir, depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    changed_depth: list[str] = []
    for stem, common, sci, _, cls_name, subtype, _ in species:
        cut_final_p = cut_final_dir / f"{stem}.png"
        canny_p = canny_dir / f"{stem}.png"
        depth_p = depth_dir / f"{stem}.png"
        genai_p = ARTIFACT_DIR / f"genai_u10_{stem}.png"

        if genai_p.exists():
            cut_im = normalize_repaired_cutout(Image.open(genai_p))
            cut_im.save(cut_final_p)
            if subtype == "bird_perched":
                meta[stem]["has_perch_stub"] = True
            changed_depth.append(stem)
        elif not cut_final_p.exists():
            cut_im = Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB")
            cut_im.save(cut_final_p)
            changed_depth.append(stem)
        else:
            cut_im = Image.open(cut_final_p).convert("RGB")
            if not depth_p.exists():
                changed_depth.append(stem)

        if not canny_p.exists() or genai_p.exists():
            build_canny_from_repaired(cut_im).save(canny_p)

    (WORK / "unseen10_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if changed_depth:
        print(f"Computing Depth-Anything-V2 maps for {len(changed_depth)} species...", flush=True)
        depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if torch.cuda.is_available() else -1,
        )
        for stem in changed_depth:
            cut_im = Image.open(cut_final_dir / f"{stem}.png").convert("RGB")
            arr = np.array(cut_im)
            fg = np.any(arr < 246, axis=-1)
            d_im = depth_pipe(cut_im)["depth"].resize((1024, 1024), Image.Resampling.BILINEAR)
            d_arr = np.array(d_im, dtype=np.float32)
            if fg.any():
                lo, hi = np.percentile(d_arr[fg], [2, 98])
                d_norm = np.clip((d_arr - lo) / max(1e-5, hi - lo) * 215.0 + 40.0, 40.0, 255.0)
            else:
                d_norm = d_arr
            d_norm[~fg] = 0.0
            Image.fromarray(np.stack([d_norm.astype(np.uint8)] * 3, axis=-1)).save(depth_dir / f"{stem}.png")
        del depth_pipe
        gc.collect()
        torch.cuda.empty_cache()

    items = []
    for stem, common, sci, _, cls_name, subtype, _ in species:
        is_perched = bool(meta.get(stem, {}).get("has_perch_stub", False))
        items.append(
            {
                "stem": stem,
                "common": common,
                "sci": sci,
                "cls_name": cls_name,
                "subtype": subtype,
                "is_perched": is_perched,
                "photo_p": WORK / "photos_selected" / f"{stem}.png",
                "cut_p": cut_final_dir / f"{stem}.png",
                "canny_p": canny_dir / f"{stem}.png",
                "depth_p": depth_dir / f"{stem}.png",
            }
        )
    return items


def run_phase_3_stylize(force_stems: set[str], unet_only_stems: set[str]) -> None:
    from diffusers import (
        ControlNetModel,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

    species = load_resolved_manifest()
    meta = json.loads((WORK / "unseen10_metadata.json").read_text(encoding="utf-8"))
    unload_ollama_model("gemma3:27b")
    torch.cuda.empty_cache()

    items = prepare_final_cutouts_and_conditions(species, meta)
    raw_dir = WORK / "raw_stylized"
    clean_dir = WORK / "clean_out"
    eq_dir = WORK / "eq_out"
    for d in (raw_dir, clean_dir, eq_dir):
        d.mkdir(parents=True, exist_ok=True)

    need_render = [
        it for it in items if (not (raw_dir / f"{it['stem']}.png").exists()) or (it["stem"] in force_stems)
    ]
    if need_render:
        print("Loading SDXL + Canny/Depth MultiControlNet + animl_style_v7-step00002000...", flush=True)
        cnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0-small",
            torch_dtype=torch.bfloat16,
            variant="fp16",
        )
        cnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            torch_dtype=torch.bfloat16,
            variant="fp16",
        )
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
            controlnet=MultiControlNetModel([cnet_canny, cnet_depth]),
            torch_dtype=torch.bfloat16,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        auto_unet = lambda it: (it["stem"] in unet_only_stems) or is_batoidea_ray_v7(it["common"], it["cls_name"])
        full_items = [it for it in need_render if not auto_unet(it)]
        unet_items = [it for it in need_render if auto_unet(it)]

        for pass_label, weight_name, pass_items in [
            ("full v7-2000", "animl_style_v7-step00002000.safetensors", full_items),
            ("unet_only v7-2000", "animl_style_v7_unet_only.safetensors", unet_items),
        ]:
            if not pass_items:
                continue
            pipe.unload_lora_weights()
            print(f"\n[{pass_label}] Loading {weight_name} for {len(pass_items)} species...", flush=True)
            pipe.load_lora_weights(str(LORA / "output"), weight_name=weight_name)
            for it in pass_items:
                stem = it["stem"]
                m = meta[stem]
                traits = m["visual_traits"]
                is_ray = is_batoidea_ray_v7(it["common"], it["cls_name"])
                if stem in unet_only_stems or is_ray:
                    traits = strip_head_eye_clauses(traits)
                prompt, neg_prompt, strength = build_prompt_and_neg_v7(
                    it["common"],
                    it["sci"],
                    it["cls_name"],
                    it["subtype"],
                    traits,
                    m.get("eye_color_in_photo", ""),
                    it["is_perched"],
                )
                canny_scale = (
                    0.54
                    if it["subtype"] == "shark_ray"
                    else (
                        0.48
                        if any(k in stem for k in ("skink", "salamander", "killdeer", "hawk"))
                        else 0.45
                    )
                )
                print(
                    f"  Rendering {stem} ({it['common']}) | perched={it['is_perched']} strength={strength:.2f} canny={canny_scale:.2f}",
                    flush=True,
                )
                cut_im = Image.open(it["cut_p"]).convert("RGB")
                canny_im = Image.open(it["canny_p"]).convert("RGB")
                depth_im = Image.open(it["depth_p"]).convert("RGB")
                cond_cut_im, cond_canny_im = prepare_conditioned_cutout_and_canny_v7(
                    cut_im, it["cls_name"], traits, base_canny=canny_im
                )

                pos_e, neg_e, pos_p, neg_p = encode_prompt_sdxl_multichunk(pipe, prompt, neg_prompt, max_chunks=3)
                gen = torch.Generator(device="cuda").manual_seed(42)
                out_im = pipe(
                    prompt_embeds=pos_e,
                    negative_prompt_embeds=neg_e,
                    pooled_prompt_embeds=pos_p,
                    negative_pooled_prompt_embeds=neg_p,
                    image=cond_cut_im,
                    control_image=[cond_canny_im, depth_im],
                    strength=strength,
                    controlnet_conditioning_scale=[canny_scale, 0.25],
                    control_guidance_start=0.0,
                    control_guidance_end=[0.76, 0.60],
                    guidance_scale=6.0,
                    num_inference_steps=36,
                    generator=gen,
                ).images[0]
                out_im.save(raw_dir / f"{stem}.png")

        del pipe, cnet_canny, cnet_depth
        gc.collect()
        torch.cuda.empty_cache()
        print("Unloaded SDXL + MultiControlNet from VRAM.", flush=True)

    rows = []
    print("\n=== Post-Clean (`clean_post_v7`) + v3 Outline Equalizer (Unseen 10) ===", flush=True)
    for it in items:
        stem = it["stem"]
        raw_im = Image.open(raw_dir / f"{stem}.png").convert("RGB")
        cut_im = Image.open(it["cut_p"]).convert("RGB")
        clean_p = clean_dir / f"{stem}.png"
        eq_p = eq_dir / f"{stem}.png"

        clean_post_v7(
            raw_im,
            cut_im,
            clean_p,
            is_perched=it["is_perched"],
            preserve_thin_appendages=(it["subtype"] == "shark_ray"),
        )
        eq_rgb, eq_stats = equalize_outline_smooth_tapered(clean_p, strength=0.90)
        eq_rgb[:20, :] = 255
        eq_rgb[-20:, :] = 255
        eq_rgb[:, :20] = 255
        eq_rgb[:, -20:] = 255
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
        rows.append(row)
        print(
            f"  {stem:26s} | AA={aa_pct:5.1f}% Gap={gap_pct:4.1f}% "
            f"Blk={reb['black_pct']:5.1f}% Tot={reb['total_pct']:5.1f}% W={sh['width_px']:4.2f}",
            flush=True,
        )

    summary = {
        "mean_aa_pct": round(float(np.mean([r["aa_midtone_pct"] for r in rows])), 2),
        "mean_gap_pct": round(float(np.mean([r["trapped_fur_gap_pct"] for r in rows])), 2),
        "mean_black_pct": round(float(np.mean([r["black_line_pct"] for r in rows])), 2),
        "mean_total_pct": round(float(np.mean([r["total_line_pct"] for r in rows])), 2),
        "mean_width_px": round(float(np.mean([r["width_px"] for r in rows])), 2),
        "mean_int_px": round(float(np.mean([r["int_crease_px"] for r in rows])), 1),
    }
    print(
        f"\nMean Unseen-10 Metrics: AA={summary['mean_aa_pct']}% Gap={summary['mean_gap_pct']}% "
        f"Blk={summary['mean_black_pct']}% Tot={summary['mean_total_pct']}% W={summary['mean_width_px']}px",
        flush=True,
    )
    (WORK / "unseen10_metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    # Render two 4-column visual boards:
    # 1. Selected Ref Photo | 2. Input Cutout | 3. Raw v7 (Pre-Eq) | 4. Final v7 (step 2000) + v3 Eq
    font_hdr = load_font(20, bold=True)
    font_sub = load_font(15, bold=True)
    font_sm = load_font(12, bold=False)
    cw, ch, hh = 380, 400, 64
    headers = [
        "1. Selected Ref Photo",
        "2. Input Cutout",
        "3. Raw v7 (Pre-Eq)",
        "4. Final v7 (step 2000) + v3 Eq",
    ]
    for part_idx, sub_items in [(1, items[:5]), (2, items[5:10])]:
        board = Image.new("RGB", (4 * cw, hh + len(sub_items) * ch), (246, 247, 249))
        draw = ImageDraw.Draw(board)
        draw.rectangle([0, 0, 4 * cw, hh], fill=(24, 29, 38))
        draw.text(
            (14, 7),
            f"animl_style_v7 (step 2000) Unseen 10 Evaluation Part {part_idx} (Zero Overlap with v6 / v7 / Prior Tests)",
            fill=(255, 255, 255),
            font=font_hdr,
        )
        for ci, htxt in enumerate(headers):
            draw.text((ci * cw + 14, 38), htxt, fill=(196, 212, 232), font=font_sub)

        for ri, it in enumerate(sub_items):
            stem = it["stem"]
            y0 = hh + ri * ch
            m_row = next(r for r in rows if r["stem"] == stem)
            col_paths = [
                it["photo_p"],
                it["cut_p"],
                raw_dir / f"{stem}.png",
                eq_dir / f"{stem}.png",
            ]
            for ci, p in enumerate(col_paths):
                x0 = ci * cw
                tile = Image.open(p).convert("RGB").resize((cw - 16, cw - 16), Image.Resampling.LANCZOS)
                board.paste(tile, (x0 + 8, y0 + 28))
                draw.rectangle([x0, y0, x0 + cw, y0 + ch], outline=(218, 222, 228), width=1)
                if ci == 0:
                    lbl = f"{it['common']} ({it['cls_name']})"
                elif ci == 1:
                    lbl = "Pre-stylization Cutout"
                elif ci == 2:
                    lbl = "Raw v7-2000 (Pre-Equalizer)"
                else:
                    lbl = f"Final v7-2000  Blk={m_row['black_line_pct']:.0f}% AA={m_row['aa_midtone_pct']:.0f}%"
                draw.text((x0 + 10, y0 + 6), lbl[:48], fill=(32, 36, 44), font=font_sm)

        out_p = ARTIFACT_DIR / f"v7_unseen10_part{part_idx}.jpg"
        board.save(out_p, quality=93)
        print(f"Saved Unseen-10 Part {part_idx} board: {out_p}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1a", "1b", "stylize"])
    parser.add_argument("--swap", nargs=2, metavar=("STEM", "CAND_RANK"))
    parser.add_argument("--force", nargs="*", default=[])
    parser.add_argument("--unet-only-stems", nargs="*", default=[])
    args = parser.parse_args()

    if args.swap:
        run_swap_candidate(args.swap[0], int(args.swap[1]))
    elif args.phase == "1a":
        run_phase_1a(force_species=set(args.force))
    elif args.phase == "1b":
        run_phase_1b(force_revet_stems=set(args.force))
    elif args.phase == "stylize":
        run_phase_3_stylize(force_stems=set(args.force), unet_only_stems=set(args.unet_only_stems))
    else:
        parser.error("Provide --phase {1a,1b,stylize} or --swap STEM CAND_RANK")


if __name__ == "__main__":
    main()
