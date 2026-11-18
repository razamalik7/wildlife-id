"""animl_style_v7 Checkpoint Shootout (`v6-1250` vs `v7-1250 / 1500 / 1750 / 2000`).

Evaluates 5 LoRA checkpoints side-by-side in our production Dual ControlNet (Canny + Depth)
+ SDXL img2img + v3 Smooth Outline Equalizer pipeline across 10 strictly held-out (unseen)
species spanning all 5 vertebrate classes.

Outputs (saved in `art-pipeline/batch_run_v7_shootout/` and artifact directory):
- `v7_shootout_part1_mammals_birds.jpg` (Mammals & Perched Birds, post-equalizer)
- `v7_shootout_part2_herps_marine.jpg` (Snake, Lizard, Turtle, Frog, Shark, post-equalizer)
- `v7_shootout_raw_pre_eq.jpg` (All 10 species x 5 checkpoints before outline equalizer)
- `shootout_metrics.json` (AA%, TrappedFurGap%, BlackLine%, TotalLine%, IntPx per checkpoint)
"""

import gc
import json
import os
import re
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

sys.path.insert(0, str(Path(__file__).parent))
from measure_outline_trough import analyze_rebound_outline
from smooth_outline_equalizer import (
    equalize_outline_smooth_tapered,
    measure_aa_midtone_pct,
    measure_trapped_fur_gap,
)
from test_fresh_20_batch2 import sanitize_species_subject
from test_fresh_20_e2e import (
    NEG_V6_UNIFIED,
    STYLE_V6,
    clean_post,
    encode_prompt_sdxl_multichunk,
)
from test_pre_stylization_crop_repair import build_canny_from_repaired
from verify_width_and_shells import measure_shells_and_width

LORA = Path(__file__).parent / "lora"
WORK_B1 = Path(__file__).parent / "batch_run_fresh_20_e2e"
WORK_B2 = Path(__file__).parent / "batch_run_fresh_20_batch2"
OUT_ROOT = Path(__file__).parent / "batch_run_v7_shootout"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

CHECKPOINTS = [
    ("v6_1250", "v6 (step 1250)", "animl_style_v6-step00001250.safetensors"),
    ("v7_1250", "v7 (step 1250)", "animl_style_v7-step00001250.safetensors"),
    ("v7_1500", "v7 (step 1500)", "animl_style_v7-step00001500.safetensors"),
    ("v7_1750", "v7 (step 1750)", "animl_style_v7-step00001750.safetensors"),
    ("v7_2000", "v7 (step 2000)", "animl_style_v7-step00002000.safetensors"),
]

# 10 strictly held-out species (0 overlap with dataset_v7):
# (stem, common, sci, cls_name, subtype, src_batch, is_perched)
SHOOTOUT_10 = [
    ("01_collared_peccary", "Collared Peccary", "Pecari tajacu", "Mammalia", "mammal_ground", "b2", False),
    ("03_gunnisons_prairie_dog", "Gunnison's Prairie Dog", "Cynomys gunnisoni", "Mammalia", "mammal_ground", "b2", False),
    ("01_desert_cottontail", "Desert Cottontail", "Sylvilagus audubonii", "Mammalia", "mammal_ground", "b1", False),
    ("06_painted_bunting", "Painted Bunting", "Passerina ciris", "Aves", "bird_perched", "b2", True),
    ("07_cedar_waxwing", "Cedar Waxwing", "Bombycilla cedrorum", "Aves", "bird_perched", "b2", True),
    ("11_ring_necked_snake", "Ring-necked Snake", "Diadophis punctatus", "Reptilia", "snake", "b1", False),
    ("13_texas_horned_lizard", "Texas Horned Lizard", "Phrynosoma cornutum", "Reptilia", "lizard", "b2", False),
    ("15_diamondback_terrapin", "Diamondback Terrapin", "Malaclemys terrapin", "Reptilia", "turtle", "b2", False),
    ("16_northern_leopard_frog", "Northern Leopard Frog", "Lithobates pipiens", "Amphibia", "amphibian", "b2", False),
    ("19_horn_shark", "Horn Shark", "Heterodontus francisci", "Elasmobranchii", "shark_ray", "b2", False),
]

RING_NECKED_SNAKE_TRAITS = (
    "dark slate-gray body, thin orange neck ring behind head, "
    "orange and black speckled ventral edge along lower coil, smooth scales, slender coiled body"
)


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


def prepare_inputs() -> list[dict]:
    meta_b1 = json.loads((WORK_B1 / "fresh_20_metadata.json").read_text(encoding="utf-8"))
    meta_b2 = json.loads((WORK_B2 / "batch2_metadata.json").read_text(encoding="utf-8"))

    canny_dir = OUT_ROOT / "canny"
    canny_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict] = []
    for stem, common, sci, cls_name, subtype, src_batch, is_perched in SHOOTOUT_10:
        if src_batch == "b2":
            cut_p = WORK_B2 / "cutouts_final" / f"{stem}.png"
            depth_p = WORK_B2 / "depth_final" / f"{stem}.png"
            meta = meta_b2[stem]
        else:
            cut_p = WORK_B1 / "cutouts_1024" / f"{stem}.png"
            depth_p = WORK_B1 / "depth" / f"{stem}.png"
            meta = meta_b1[stem]

        if not cut_p.exists():
            raise FileNotFoundError(f"Missing cutout: {cut_p}")
        if not depth_p.exists():
            raise FileNotFoundError(f"Missing depth map: {depth_p}")

        cut_im = Image.open(cut_p).convert("RGB")
        canny_p = canny_dir / f"{stem}.png"
        if not canny_p.exists():
            build_canny_from_repaired(cut_im).save(canny_p)

        subj_phrase, extra_neg = sanitize_species_subject(common, sci, cls_name, subtype)
        traits = RING_NECKED_SNAKE_TRAITS if stem == "11_ring_necked_snake" else meta["visual_traits"]
        eye_col = meta.get("eye_color_in_photo", "")
        eye_clause = f", {eye_col}" if eye_col and "natural dark" not in eye_col.lower() else ""

        neg_prompt = extra_neg + NEG_V6_UNIFIED
        perch_clause = ""
        if is_perched:
            perch_clause = ", perched with both feet gripping a short natural wooden twig stub strictly under its feet"
            neg_prompt = re.sub(r"\b(branch|twig|wood),\s*", "", neg_prompt)

        prompt = (
            f"animl_gouache_style, {subj_phrase}, full body wildlife illustration{perch_clause}, "
            f"single animal, {traits}{eye_clause}, {STYLE_V6}"
        )
        canny_scale = (
            0.48
            if any(k in stem for k in ("shark", "ray", "salamander", "frog", "terrapin", "gecko", "bunting"))
            else 0.45
        )
        prepared.append(
            {
                "stem": stem,
                "common": common,
                "sci": sci,
                "cls_name": cls_name,
                "subtype": subtype,
                "is_perched": is_perched,
                "cut_p": cut_p,
                "canny_p": canny_p,
                "depth_p": depth_p,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "canny_scale": canny_scale,
                "depth_scale": 0.25,
            }
        )
    return prepared


def run_sdxl_pass(items: list[dict]) -> None:
    need_any = any(
        not (OUT_ROOT / "raw_stylized" / ckpt_tag / f"{it['stem']}.png").exists()
        for ckpt_tag, _, _ in CHECKPOINTS
        for it in items
    )
    if not need_any:
        print("Pass 1 (SDXL + MultiControlNet) already cached for all 50 renders.", flush=True)
        return

    print("Loading SDXL + Canny & Depth ControlNets in bfloat16...", flush=True)
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

    lora_loaded = False
    for ckpt_tag, ckpt_label, weight_name in CHECKPOINTS:
        raw_dir = OUT_ROOT / "raw_stylized" / ckpt_tag
        raw_dir.mkdir(parents=True, exist_ok=True)
        missing = [it for it in items if not (raw_dir / f"{it['stem']}.png").exists()]
        if not missing:
            print(f"[{ckpt_label}] All 10 raw renders already cached.", flush=True)
            continue

        if lora_loaded:
            pipe.unload_lora_weights()
        print(f"\n[{ckpt_label}] Loading LoRA {weight_name}...", flush=True)
        pipe.load_lora_weights(str(LORA / "output"), weight_name=weight_name)
        lora_loaded = True

        for idx, it in enumerate(items):
            raw_p = raw_dir / f"{it['stem']}.png"
            if raw_p.exists():
                continue
            print(f"  [{ckpt_label}] ({idx + 1}/10) {it['stem']} ({it['common']})...", flush=True)
            cut_im = Image.open(it["cut_p"]).convert("RGB")
            canny_im = Image.open(it["canny_p"]).convert("RGB")
            depth_im = Image.open(it["depth_p"]).convert("RGB")

            pos_e, neg_e, pos_p, neg_p = encode_prompt_sdxl_multichunk(
                pipe, it["prompt"], it["neg_prompt"], max_chunks=3
            )
            gen = torch.Generator(device="cuda").manual_seed(42)
            out_im = pipe(
                prompt_embeds=pos_e,
                negative_prompt_embeds=neg_e,
                pooled_prompt_embeds=pos_p,
                negative_pooled_prompt_embeds=neg_p,
                image=cut_im,
                control_image=[canny_im, depth_im],
                strength=0.54,
                controlnet_conditioning_scale=[it["canny_scale"], it["depth_scale"]],
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
    print("Unloaded SDXL + MultiControlNet from VRAM before BiRefNet cleanup.", flush=True)


def run_post_and_metrics(items: list[dict]) -> dict:
    per_species: list[dict] = []
    agg: dict[str, list[dict]] = {tag: [] for tag, _, _ in CHECKPOINTS}

    for ckpt_tag, ckpt_label, weight_name in CHECKPOINTS:
        raw_dir = OUT_ROOT / "raw_stylized" / ckpt_tag
        clean_dir = OUT_ROOT / "clean_out" / ckpt_tag
        eq_dir = OUT_ROOT / "eq_out" / ckpt_tag
        clean_dir.mkdir(parents=True, exist_ok=True)
        eq_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Post-Clean + v3 Outline Equalizer: {ckpt_label} ===", flush=True)
        for it in items:
            stem = it["stem"]
            raw_p = raw_dir / f"{stem}.png"
            clean_p = clean_dir / f"{stem}.png"
            eq_p = eq_dir / f"{stem}.png"

            if not clean_p.exists():
                res_im = Image.open(raw_p).convert("RGB")
                clean_post(res_im, clean_p)
                if it["is_perched"]:
                    post_rgba = Image.open(clean_p).convert("RGBA")
                    post_arr = np.array(post_rgba)
                    cut_arr = np.array(Image.open(it["cut_p"]).convert("RGB"))
                    fg_cut = np.any(cut_arr < 242, axis=-1)
                    fg_eroded = cv2.erode(fg_cut.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
                    res_arr = np.array(res_im)
                    missing_perch = fg_eroded & (post_arr[:, :, 3] < 128)
                    post_arr[missing_perch, :3] = res_arr[missing_perch]
                    post_arr[missing_perch, 3] = 255
                    # Flatten cleanly to RGB on pure white so clean_out shows exact isolated cutout
                    bg = Image.new("RGB", post_rgba.size, (255, 255, 255))
                    rgba_fixed = Image.fromarray(post_arr)
                    bg.paste(rgba_fixed, mask=rgba_fixed.split()[3])
                    bg.save(clean_p)

            if not eq_p.exists():
                eq_rgb, eq_stats = equalize_outline_smooth_tapered(clean_p, strength=0.90)
                eq_rgb[:20, :] = 255
                eq_rgb[-20:, :] = 255
                eq_rgb[:, :20] = 255
                eq_rgb[:, -20:] = 255
                Image.fromarray(eq_rgb).save(eq_p)
            else:
                eq_rgb = np.array(Image.open(eq_p).convert("RGB"))
                _, eq_stats = equalize_outline_smooth_tapered(clean_p, strength=0.90)

            aa_pct = measure_aa_midtone_pct(eq_rgb)
            gap_pct = measure_trapped_fur_gap(eq_rgb)
            reb = analyze_rebound_outline(eq_p)
            sh = measure_shells_and_width(eq_p)

            row = {
                "ckpt_tag": ckpt_tag,
                "ckpt_label": ckpt_label,
                "stem": stem,
                "common": it["common"],
                "aa_midtone_pct": round(aa_pct, 2),
                "trapped_fur_gap_pct": round(gap_pct, 2),
                "black_line_pct": round(reb["black_pct"], 2),
                "total_line_pct": round(reb["total_pct"], 2),
                "width_px": round(sh["width_px"], 2),
                "rim3_l": round(sh["rim3_l"], 2),
                "soft_interior": round(eq_stats["soft_interior"], 2),
                "int_crease_px": int(eq_stats["int_crease_px"]),
            }
            per_species.append(row)
            agg[ckpt_tag].append(row)
            print(
                f"  {stem:26s} | AA={aa_pct:5.1f}% Gap={gap_pct:4.1f}% "
                f"Blk={reb['black_pct']:5.1f}% Tot={reb['total_pct']:5.1f}% "
                f"W={sh['width_px']:4.2f} IntPx={eq_stats['int_crease_px']:5d}",
                flush=True,
            )

    summary = {}
    print("\n=== AGGREGATE 10-SPECIES CHECKPOINT SUMMARY (Post-Equalizer) ===")
    print(f"{'Checkpoint':16s} | {'AA%':>7s} | {'Gap%':>6s} | {'Blk%':>6s} | {'Tot%':>6s} | {'Width':>6s} | {'IntPx':>7s}")
    print("-" * 74)
    for ckpt_tag, ckpt_label, weight_name in CHECKPOINTS:
        rows = agg[ckpt_tag]
        s = {
            "label": ckpt_label,
            "weight_name": weight_name,
            "mean_aa_pct": round(float(np.mean([r["aa_midtone_pct"] for r in rows])), 2),
            "mean_gap_pct": round(float(np.mean([r["trapped_fur_gap_pct"] for r in rows])), 2),
            "mean_black_pct": round(float(np.mean([r["black_line_pct"] for r in rows])), 2),
            "mean_total_pct": round(float(np.mean([r["total_line_pct"] for r in rows])), 2),
            "mean_width_px": round(float(np.mean([r["width_px"] for r in rows])), 2),
            "mean_int_px": round(float(np.mean([r["int_crease_px"] for r in rows])), 1),
        }
        summary[ckpt_tag] = s
        print(
            f"{ckpt_label:16s} | {s['mean_aa_pct']:6.1f}% | {s['mean_gap_pct']:5.1f}% | "
            f"{s['mean_black_pct']:5.1f}% | {s['mean_total_pct']:5.1f}% | "
            f"{s['mean_width_px']:5.2f} | {s['mean_int_px']:7.1f}"
        )

    metrics_payload = {"summary": summary, "per_species": per_species}
    (OUT_ROOT / "shootout_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def render_boards(items: list[dict], metrics_payload: dict) -> None:
    font_hdr = load_font(20, bold=True)
    font_sub = load_font(15, bold=True)
    font_sm = load_font(12, bold=False)

    # Boards 1 & 2: Post-Equalizer comparison (6 columns: Input Cutout + 5 checkpoints)
    splits = [
        ("v7_shootout_part1_mammals_birds.jpg", "animl_style_v7 Shootout Part 1: Held-Out Mammals & Perched Birds (Dual ControlNet + v3 Equalizer)", items[:5]),
        ("v7_shootout_part2_herps_marine.jpg", "animl_style_v7 Shootout Part 2: Held-Out Snake, Lizard, Turtle, Frog & Shark (Dual ControlNet + v3 Equalizer)", items[5:10]),
    ]
    cw, ch, hh = 360, 380, 64
    cols = 6
    headers = ["0. Input Cutout"] + [lbl for _, lbl, _ in CHECKPOINTS]

    for fname, title, sub_items in splits:
        board = Image.new("RGB", (cols * cw, hh + len(sub_items) * ch), (246, 247, 249))
        draw = ImageDraw.Draw(board)
        draw.rectangle([0, 0, cols * cw, hh], fill=(24, 29, 38))
        draw.text((14, 7), title, fill=(255, 255, 255), font=font_hdr)
        for ci, htxt in enumerate(headers):
            draw.text((ci * cw + 14, 38), htxt, fill=(196, 212, 232), font=font_sub)

        for ri, it in enumerate(sub_items):
            y0 = hh + ri * ch
            stem = it["stem"]
            paths = [it["cut_p"]] + [OUT_ROOT / "eq_out" / tag / f"{stem}.png" for tag, _, _ in CHECKPOINTS]
            for ci, p in enumerate(paths):
                x0 = ci * cw
                tile = Image.open(p).convert("RGB").resize((cw - 16, cw - 16), Image.Resampling.LANCZOS)
                board.paste(tile, (x0 + 8, y0 + 28))
                draw.rectangle([x0, y0, x0 + cw, y0 + ch], outline=(218, 222, 228), width=1)
                if ci == 0:
                    lbl = f"{it['common']} ({it['cls_name']})"
                else:
                    tag = CHECKPOINTS[ci - 1][0]
                    row_m = next(
                        r for r in metrics_payload["per_species"] if r["ckpt_tag"] == tag and r["stem"] == stem
                    )
                    lbl = f"{headers[ci]}  Blk={row_m['black_line_pct']:.0f}% AA={row_m['aa_midtone_pct']:.0f}%"
                draw.text((x0 + 10, y0 + 6), lbl[:44], fill=(32, 36, 44), font=font_sm)

        out_p = ARTIFACT_DIR / fname
        board.save(out_p, quality=93)
        print(f"Saved comparison board: {out_p}", flush=True)

    # Board 3: Pre-Equalizer (`clean_out`) 10x5 comparison so native LoRA ink/color/shading are directly visible
    cw3, ch3, hh3 = 320, 340, 62
    cols3 = len(CHECKPOINTS)
    raw_board = Image.new("RGB", (cols3 * cw3, hh3 + len(items) * ch3), (246, 247, 249))
    draw3 = ImageDraw.Draw(raw_board)
    draw3.rectangle([0, 0, cols3 * cw3, hh3], fill=(24, 29, 38))
    draw3.text(
        (14, 7),
        "animl_style_v7 Raw Pre-Equalizer Comparison (Native LoRA Shading, Colors & Ink Before v3 Equalizer)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    for ci, (_, ckpt_label, _) in enumerate(CHECKPOINTS):
        draw3.text((ci * cw3 + 14, 36), ckpt_label, fill=(196, 212, 232), font=font_sub)

    for ri, it in enumerate(items):
        y0 = hh3 + ri * ch3
        stem = it["stem"]
        for ci, (tag, ckpt_label, _) in enumerate(CHECKPOINTS):
            x0 = ci * cw3
            p = OUT_ROOT / "clean_out" / tag / f"{stem}.png"
            tile = Image.open(p).convert("RGB").resize((cw3 - 16, cw3 - 16), Image.Resampling.LANCZOS)
            raw_board.paste(tile, (x0 + 8, y0 + 26))
            draw3.rectangle([x0, y0, x0 + cw3, y0 + ch3], outline=(218, 222, 228), width=1)
            lbl = f"{it['common']} | {ckpt_label}"
            draw3.text((x0 + 10, y0 + 5), lbl[:42], fill=(32, 36, 44), font=font_sm)

    raw_board_p = ARTIFACT_DIR / "v7_shootout_raw_pre_eq.jpg"
    raw_board.save(raw_board_p, quality=92)
    print(f"Saved pre-equalizer board: {raw_board_p}", flush=True)


def main() -> None:
    items = prepare_inputs()
    run_sdxl_pass(items)
    metrics_payload = run_post_and_metrics(items)
    render_boards(items, metrics_payload)


if __name__ == "__main__":
    main()
