"""Re-run Board 1 (5 held-out mammals & perched birds) with the 3 general v7 pipeline fixes.

Compares side-by-side in `v7_board1_fixes_comparison.jpg`:
  Col 1: Input Cutout
  Col 2: v6 (step 1250)  [from batch_run_v7_shootout]
  Col 3: Old v7 (step 2000) [from batch_run_v7_shootout]
  Col 4: Fixed v7 (step 2000) [upgraded Wiki + Gemma3:27b traits + prompt guards + clean_post_v7]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from measure_outline_trough import analyze_rebound_outline
from pipeline_v7 import (
    build_prompt_and_neg_v7,
    clean_post_v7,
    fetch_wiki_clauses_map,
    prepare_conditioned_cutout_and_canny_v7,
    vet_side_by_side_and_extract_traits_v7,
)
from smooth_outline_equalizer import (
    equalize_outline_smooth_tapered,
    measure_aa_midtone_pct,
    measure_trapped_fur_gap,
)
from test_fresh_20_e2e import encode_prompt_sdxl_multichunk, unload_ollama_model
from test_pre_stylization_crop_repair import build_canny_from_repaired
from verify_width_and_shells import measure_shells_and_width

ROOT = Path(__file__).parent
LORA = ROOT / "lora"
WORK_B1 = ROOT / "batch_run_fresh_20_e2e"
WORK_B2 = ROOT / "batch_run_fresh_20_batch2"
SHOOTOUT_DIR = ROOT / "batch_run_v7_shootout"
OUT_ROOT = ROOT / "batch_run_v7_board1_fixed"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

BOARD1_SPECIES = [
    ("01_collared_peccary", "Collared Peccary", "Pecari tajacu", "Mammalia", "mammal_ground", "b2", False),
    ("03_gunnisons_prairie_dog", "Gunnison's Prairie Dog", "Cynomys gunnisoni", "Mammalia", "mammal_ground", "b2", False),
    ("01_desert_cottontail", "Desert Cottontail", "Sylvilagus audubonii", "Mammalia", "mammal_ground", "b1", False),
    ("06_painted_bunting", "Painted Bunting", "Passerina ciris", "Aves", "bird_perched", "b2", True),
    ("07_cedar_waxwing", "Cedar Waxwing", "Bombycilla cedrorum", "Aves", "bird_perched", "b2", True),
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


def resolve_paths() -> list[dict]:
    items = []
    canny_cache = OUT_ROOT / "canny"
    canny_cache.mkdir(parents=True, exist_ok=True)
    for stem, common, sci, cls_name, subtype, src_batch, is_perched in BOARD1_SPECIES:
        if src_batch == "b2":
            photo_p = WORK_B2 / "photos_selected" / f"{stem}.png"
            cut_p = WORK_B2 / "cutouts_final" / f"{stem}.png"
            depth_p = WORK_B2 / "depth_final" / f"{stem}.png"
        else:
            photo_p = WORK_B1 / "photos_selected" / f"{stem}.png"
            cut_p = WORK_B1 / "cutouts_1024" / f"{stem}.png"
            depth_p = WORK_B1 / "depth" / f"{stem}.png"
        canny_p = SHOOTOUT_DIR / "canny" / f"{stem}.png"
        if not canny_p.exists():
            canny_p = canny_cache / f"{stem}.png"
            build_canny_from_repaired(Image.open(cut_p).convert("RGB")).save(canny_p)
        items.append(
            {
                "stem": stem,
                "common": common,
                "sci": sci,
                "cls_name": cls_name,
                "subtype": subtype,
                "src_batch": src_batch,
                "is_perched": is_perched,
                "photo_p": photo_p,
                "cut_p": cut_p,
                "canny_p": canny_p,
                "depth_p": depth_p,
            }
        )
    return items


def run_phase_a_traits(
    items: list[dict],
    force_refresh: bool = False,
    refresh_stems: set[str] | None = None,
) -> dict:
    meta_path = OUT_ROOT / "board1_fixed_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    target_refresh = refresh_stems or set()
    need_items = [
        it for it in items if force_refresh or (it["stem"] not in meta) or (it["stem"] in target_refresh)
    ]
    if not need_items:
        print("Phase A: using cached board1_fixed_metadata.json", flush=True)
        return meta

    print(f"Phase A: Fetching upgraded Wikipedia clauses & Gemma3:27b traits for {len(need_items)} species...", flush=True)
    wiki_map = fetch_wiki_clauses_map([(it["common"], it["sci"]) for it in need_items])
    for it in need_items:
        stem = it["stem"]
        pad_photo = Image.open(it["photo_p"]).convert("RGB")
        cut_im = Image.open(it["cut_p"]).convert("RGB")
        wiki_c = wiki_map.get(it["common"], "")
        print(f"  [Gemma3:27b] Extracting v7 traits for {stem} (wiki len={len(wiki_c)})...", flush=True)
        res = vet_side_by_side_and_extract_traits_v7(
            pad_photo, cut_im, it["common"], it["sci"], it["subtype"], wiki_c
        )
        meta[stem] = {
            "common_name": it["common"],
            "scientific_name": it["sci"],
            "class": it["cls_name"],
            "subtype": it["subtype"],
            "is_perched": it["is_perched"],
            "wiki_clauses": wiki_c,
            "eye_color_in_photo": res.get("eye_color_in_photo", ""),
            "visual_traits": res.get("visual_traits", ""),
        }
        print(f"    -> eye={meta[stem]['eye_color_in_photo']!r}", flush=True)
        print(f"    -> traits={meta[stem]['visual_traits']!r}", flush=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    unload_ollama_model("gemma3:27b")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    return meta


def run_phase_b_sdxl(
    items: list[dict],
    meta: dict,
    peccary_strength_override: float | None = None,
    peccary_canny_override: float | None = None,
    force_rerender_stems: set[str] | None = None,
) -> None:
    raw_dir = OUT_ROOT / "raw_stylized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    force_stems = force_rerender_stems or set()
    need_any = any((not (raw_dir / f"{it['stem']}.png").exists()) or (it["stem"] in force_stems) for it in items)
    if not need_any:
        print("Phase B: All 5 raw renders already exist.", flush=True)
        return

    print("Phase B: Loading SDXL + Canny/Depth MultiControlNet + animl_style_v7-step00002000...", flush=True)
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
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v7-step00002000.safetensors")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.to("cuda")

    for idx, it in enumerate(items):
        stem = it["stem"]
        raw_p = raw_dir / f"{stem}.png"
        if raw_p.exists() and stem not in force_stems:
            continue

        m = meta[stem]
        prompt, neg_prompt, strength = build_prompt_and_neg_v7(
            it["common"],
            it["sci"],
            it["cls_name"],
            it["subtype"],
            m["visual_traits"],
            m.get("eye_color_in_photo", ""),
            it["is_perched"],
        )
        canny_scale = 0.48 if "bunting" in stem else 0.45
        if stem == "01_collared_peccary":
            if peccary_strength_override is not None:
                strength = peccary_strength_override
            if peccary_canny_override is not None:
                canny_scale = peccary_canny_override

        print(f"  [Phase B {idx + 1}/5] {stem} (strength={strength:.2f}, canny={canny_scale:.2f})...", flush=True)
        print(f"    PROMPT: {prompt}", flush=True)
        cut_im = Image.open(it["cut_p"]).convert("RGB")
        canny_im = Image.open(it["canny_p"]).convert("RGB")
        depth_im = Image.open(it["depth_p"]).convert("RGB")
        cond_cut_im, cond_canny_im = prepare_conditioned_cutout_and_canny_v7(
            cut_im, it["cls_name"], m["visual_traits"], base_canny=canny_im
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
        out_im.save(raw_p)

    del pipe, cnet_canny, cnet_depth
    gc.collect()
    torch.cuda.empty_cache()
    print("Phase B complete; unloaded SDXL + MultiControlNet from VRAM.", flush=True)


def run_phase_c_post_and_board(items: list[dict]) -> dict:
    raw_dir = OUT_ROOT / "raw_stylized"
    clean_dir = OUT_ROOT / "clean_out"
    eq_dir = OUT_ROOT / "eq_out"
    clean_dir.mkdir(parents=True, exist_ok=True)
    eq_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("\n=== Phase C: clean_post_v7 + v3 Outline Equalizer ===", flush=True)
    for it in items:
        stem = it["stem"]
        raw_im = Image.open(raw_dir / f"{stem}.png").convert("RGB")
        cut_im = Image.open(it["cut_p"]).convert("RGB")
        clean_p = clean_dir / f"{stem}.png"
        eq_p = eq_dir / f"{stem}.png"

        clean_post_v7(raw_im, cut_im, clean_p, is_perched=it["is_perched"])
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

    metrics_p = OUT_ROOT / "board1_fixed_metrics.json"
    metrics_p.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    # Render 4-column comparison board: Input Cutout | v6 (1250) | Old v7 (2000) | Fixed v7 (2000)
    shoot_m = json.loads((SHOOTOUT_DIR / "shootout_metrics.json").read_text(encoding="utf-8"))["per_species"]
    cw, ch, hh = 380, 400, 64
    headers = ["1. Input Cutout", "2. v6 (step 1250)", "3. Old v7 (step 2000)", "4. Fixed v7 (step 2000)"]
    board = Image.new("RGB", (4 * cw, hh + len(items) * ch), (246, 247, 249))
    draw = ImageDraw.Draw(board)
    font_hdr = load_font(20, bold=True)
    font_sub = load_font(15, bold=True)
    font_sm = load_font(12, bold=False)

    draw.rectangle([0, 0, 4 * cw, hh], fill=(24, 29, 38))
    draw.text(
        (14, 7),
        "Board 1 Pipeline Fixes Validation: v6 (1250) vs Old v7 (2000) vs Fixed v7 (2000)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    for ci, htxt in enumerate(headers):
        draw.text((ci * cw + 14, 38), htxt, fill=(196, 212, 232), font=font_sub)

    for ri, it in enumerate(items):
        stem = it["stem"]
        y0 = hh + ri * ch
        col_paths = [
            it["cut_p"],
            SHOOTOUT_DIR / "eq_out" / "v6_1250" / f"{stem}.png",
            SHOOTOUT_DIR / "eq_out" / "v7_2000" / f"{stem}.png",
            eq_dir / f"{stem}.png",
        ]
        m_old = next((r for r in shoot_m if r["ckpt_tag"] == "v7_2000" and r["stem"] == stem), None)
        m_v6 = next((r for r in shoot_m if r["ckpt_tag"] == "v6_1250" and r["stem"] == stem), None)
        m_new = next(r for r in rows if r["stem"] == stem)
        for ci, p in enumerate(col_paths):
            x0 = ci * cw
            tile = Image.open(p).convert("RGB").resize((cw - 16, cw - 16), Image.Resampling.LANCZOS)
            board.paste(tile, (x0 + 8, y0 + 28))
            draw.rectangle([x0, y0, x0 + cw, y0 + ch], outline=(218, 222, 228), width=1)
            if ci == 0:
                lbl = f"{it['common']} ({it['cls_name']})"
            elif ci == 1 and m_v6:
                lbl = f"v6 (1250)  Blk={m_v6['black_line_pct']:.0f}% AA={m_v6['aa_midtone_pct']:.0f}%"
            elif ci == 2 and m_old:
                lbl = f"Old v7 (2000)  Blk={m_old['black_line_pct']:.0f}% AA={m_old['aa_midtone_pct']:.0f}%"
            else:
                lbl = f"Fixed v7 (2000)  Blk={m_new['black_line_pct']:.0f}% AA={m_new['aa_midtone_pct']:.0f}%"
            draw.text((x0 + 10, y0 + 6), lbl[:48], fill=(32, 36, 44), font=font_sm)

    board_p = ARTIFACT_DIR / "v7_board1_fixes_comparison.jpg"
    board.save(board_p, quality=93)
    print(f"\nSaved comparison board: {board_p}", flush=True)
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-traits", action="store_true")
    parser.add_argument("--refresh-stems", nargs="*", default=[])
    parser.add_argument("--force-rerender", nargs="*", default=[])
    parser.add_argument("--peccary-strength", type=float, default=None)
    parser.add_argument("--peccary-canny", type=float, default=None)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    items = resolve_paths()
    meta = run_phase_a_traits(items, force_refresh=args.force_traits, refresh_stems=set(args.refresh_stems))
    run_phase_b_sdxl(
        items,
        meta,
        peccary_strength_override=args.peccary_strength,
        peccary_canny_override=args.peccary_canny,
        force_rerender_stems=set(args.force_rerender),
    )
    run_phase_c_post_and_board(items)


if __name__ == "__main__":
    main()
