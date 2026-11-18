"""2x2 Ablation Experiment + Wikipedia Visual Trait Extractor for the 25 Masters.

Part 1: Run a clean 2x2 ablation on 5 unseen species at fixed checkpoint (s1000, mul 0.85, strength 0.68):
  - Col 1: Raw Cutout + Short Prompt (No Gouache Pre-Flatten, No Wiki Prompt)
  - Col 2: Gouache Pre-Flatten + Short Prompt (Gouache ONLY, No Wiki Prompt)
  - Col 3: Strong Gouache Pre-Flatten (k=8) + Short Prompt (Stronger Gouache ONLY, No Wiki)
  - Col 4: Raw Cutout + Wiki Prompt (Wiki Prompt ONLY, No Gouache Pre-Flatten)
  - Col 5: Gouache Pre-Flatten + Wiki Prompt (Both Gouache + Wiki Prompt)

Part 2: Fetch Wikipedia text for our 25 training masters + 11 test species and extract
clean field-guide visual descriptions (colors, plumage/coat/shell markings, anatomy).
"""

import json
import re
import subprocess
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
ABL_DIR = WORK / "ablation_2x2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")


def pre_flatten_gouache(src_p: Path, dst_p: Path, k_colors: int = 12, strong: bool = False) -> None:
    """Flatten a photo cutout into matte color zones."""
    im = Image.open(src_p).convert("RGB")
    arr = np.array(im).astype(np.float32)
    fg_mask = (arr < 248).any(axis=2)
    if not fg_mask.any():
        im.save(dst_p)
        return

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    fg_lum = lum[fg_mask]
    p5, p95 = np.percentile(fg_lum, 5), np.percentile(fg_lum, 95)
    if p95 > p5 + 10:
        lo, hi = (52.0, 216.0) if strong else (44.0, 222.0)
        target_lum = lo + (np.clip((fg_lum - p5) / (p95 - p5), 0.0, 1.0) ** 0.88) * (hi - lo)
        ratio = target_lum / np.maximum(fg_lum, 1.0)
        blend = 0.82 if strong else 0.65
        scale = (1.0 - blend) + blend * ratio
        for c in range(3):
            arr[:, :, c][fg_mask] = np.clip(arr[:, :, c][fg_mask] * scale, 28, 240)

    matte_im = Image.fromarray(arr.astype(np.uint8))
    if strong:
        # Stronger paint-block abstraction: 3 passes of size-7 median filter + 100% quantization into 8 colors
        smooth_im = (
            matte_im.filter(ImageFilter.MedianFilter(size=7))
            .filter(ImageFilter.MedianFilter(size=7))
            .filter(ImageFilter.MedianFilter(size=5))
        )
        s_arr = np.array(smooth_im)
        s_arr[~fg_mask] = [255, 255, 255]
        q_im = Image.fromarray(s_arr).quantize(
            colors=k_colors + 1, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        ).convert("RGB")
        final_im = q_im.filter(ImageFilter.MedianFilter(size=5))
        final_np = np.array(final_im)
        final_np[~fg_mask] = [255, 255, 255]
        Image.fromarray(final_np).save(dst_p)
    else:
        smooth_im = matte_im.filter(ImageFilter.MedianFilter(size=5)).filter(ImageFilter.MedianFilter(size=5))
        s_arr = np.array(smooth_im)
        s_arr[~fg_mask] = [255, 255, 255]
        q_im = Image.fromarray(s_arr).quantize(
            colors=k_colors + 1, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        ).convert("RGB")
        q_arr = np.array(q_im)
        final_arr = (0.75 * q_arr.astype(np.float32) + 0.25 * s_arr.astype(np.float32)).astype(np.uint8)
        final_im = Image.fromarray(final_arr).filter(ImageFilter.MedianFilter(size=3))
        final_np = np.array(final_im)
        final_np[~fg_mask] = [255, 255, 255]
        Image.fromarray(final_np).save(dst_p)


def clean_and_center(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    im = Image.open(src).convert("RGBA")
    cut = remove(im)
    box = subject_bbox(cut)
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    canvas.convert("RGB").save(dst)


def run_ablation() -> None:
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    raw_in = ABL_DIR / "in_raw"
    flat_in = ABL_DIR / "in_flat"
    strong_in = ABL_DIR / "in_strong"
    for d in (raw_in, flat_in, strong_in):
        d.mkdir(exist_ok=True)

    NEG_BASE = (
        "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
        "heavy black outline, sticker, cross-hatching, pen hatching, "
        "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
        "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
        "multiple animals, duplicate, tiling, text, watermark, border, cropped body, cut off, "
        "speckled background, confetti"
    )
    STYLE_CLAUSE = (
        "flat stylized gouache wildlife illustration, crisp distinct painted color steps, "
        "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
    )

    test_cases = [
        ("04_mule_deer", "mule_deer", "Mule Deer",
         f"a mule deer, full body wildlife illustration, three-quarter view, {STYLE_CLAUSE}",
         f"a mule deer buck, warm tan-brown coat, white throat and rump patch, black-tipped tail, large mule ears, forked antlers, standing full body with all four hooves visible, three-quarter view, {STYLE_CLAUSE}"),
        ("06_california_ground_squirrel", "california_ground_squirrel", "California Ground Squirrel",
         f"a california ground squirrel, full body wildlife illustration, three-quarter view, {STYLE_CLAUSE}",
         f"a california ground squirrel, mottled brown and grey fur with pale grey shoulder cape, buffy belly, bushy tail, standing upright with front paws tucked, three-quarter view, {STYLE_CLAUSE}"),
        ("07_american_pika", "american_pika", "American Pika",
         f"an american pika, full body wildlife illustration, three-quarter view, {STYLE_CLAUSE}",
         f"an american pika, small round-bodied mammal with soft greyish-brown and buff fur, rounded ears, no visible tail, crouching full body with four paws visible, three-quarter view, {STYLE_CLAUSE}"),
        ("08_pond_slider", "pond_slider", "Pond Slider",
         f"a pond slider, full body wildlife illustration, three-quarter view, {STYLE_CLAUSE}",
         f"a red-eared pond slider turtle, oval dark olive-green aquatic carapace without yellow spots, bright red stripe behind eye, yellow-striped green legs, swimming pose, three-quarter view, {STYLE_CLAUSE}"),
        ("03_green-winged_teal", "green-winged_teal", "Green-winged Teal",
         f"a green-winged teal, full body wildlife illustration, three-quarter view, {STYLE_CLAUSE}",
         f"a male green-winged teal duck, chestnut head with iridescent green eye-patch, speckled pinkish-buff breast, vermiculated grey flanks, standing full body with webbed feet, three-quarter view, {STYLE_CLAUSE}"),
    ]

    short_prompts = []
    wiki_prompts = []
    import shutil
    for i, (inp_stem, slug, _, p_short, p_wiki) in enumerate(test_cases):
        src_p = WORK / "inputs" / f"{inp_stem}.png"
        shutil.copyfile(src_p, raw_in / f"{i:02d}_{slug}.png")
        pre_flatten_gouache(src_p, flat_in / f"{i:02d}_{slug}.png", k_colors=12, strong=False)
        pre_flatten_gouache(src_p, strong_in / f"{i:02d}_{slug}.png", k_colors=8, strong=True)
        short_prompts.append(f"{p_short} --n {NEG_BASE} --d 42")
        wiki_prompts.append(f"{p_wiki} --n {NEG_BASE} --d 42")

    p_short_file = ABL_DIR / "prompts_short.txt"
    p_wiki_file = ABL_DIR / "prompts_wiki.txt"
    p_short_file.write_text("\n".join(short_prompts), encoding="utf-8")
    p_wiki_file.write_text("\n".join(wiki_prompts), encoding="utf-8")

    # Combine all 5 conditions into a single 25-image sdxl_gen_img call so SDXL loads into VRAM ONLY ONCE!
    all_in = ABL_DIR / "all_in"
    all_out = ABL_DIR / "all_out"
    all_clean = ABL_DIR / "all_clean"
    for d in (all_in, all_out, all_clean):
        d.mkdir(exist_ok=True)

    conditions = [
        ("c1_raw_nowiki", raw_in, short_prompts),
        ("c2_flat12_nowiki", flat_in, short_prompts),
        ("c3_flat8strong_nowiki", strong_in, short_prompts),
        ("c4_raw_wiki", raw_in, wiki_prompts),
        ("c5_flat12_wiki", flat_in, wiki_prompts),
    ]

    combined_prompts = []
    idx_counter = 0
    for c_tag, c_in_dir, c_plist in conditions:
        for i, (_, slug, _, _, _) in enumerate(test_cases):
            src = c_in_dir / f"{i:02d}_{slug}.png"
            dst = all_in / f"{idx_counter:02d}_{c_tag}__{slug}.png"
            shutil.copyfile(src, dst)
            combined_prompts.append(c_plist[i])
            idx_counter += 1

    all_pfile = ABL_DIR / "prompts_all25.txt"
    all_pfile.write_text("\n".join(combined_prompts), encoding="utf-8")

    if len(list(all_out.glob("*.png"))) < 25:
        subprocess.run([
            str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
            "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
            "--outdir", str(all_out),
            "--network_module", "networks.lora",
            "--network_weights", str(LORA / "output" / "animl_style_v4-step00001000.safetensors"),
            "--network_mul", "0.85",
            "--max_embeddings_multiples", "3",
            "--from_file", str(all_pfile),
            "--image_path", str(all_in),
            "--use_original_file_name",
            "--strength", "0.66",
            "--W", "1024", "--H", "1024",
            "--steps", "26", "--sampler", "euler_a",
            "--scale", "7.0", "--bf16", "--sdpa",
            "--batch_size", "1", "--images_per_prompt", "1",
        ], check=True)

    for p in all_out.glob("*.png"):
        clean_and_center(p, all_clean / p.name)

    # Build the 2x2 (+ strong gouache) Ablation Sheet
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
        font_sm = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    cols = [
        "1. Baseline (No Gouache, No Wiki)",
        "2. Gouache-12 Only (No Wiki)",
        "3. Strong Gouache-8 Only (No Wiki)",
        "4. Wiki Only (No Gouache)",
        "5. Both (Gouache-12 + Wiki)",
    ]
    cw, ch = 340, 340
    W = len(cols) * cw
    H = 48 + len(test_cases) * (ch + 32)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 10, 12), t, fill=(20, 20, 20), font=font)

    for ri, (_, slug, common, _, _) in enumerate(test_cases):
        y_top = 48 + ri * (ch + 32)
        d.rectangle([0, y_top, W, y_top + 32], fill=(232, 232, 236))
        d.text((14, y_top + 6), f"{common} ({slug}) — Isolated 2x2 Ablation @ Fixed Step 1000, mul 0.85, str 0.66", fill=(10, 10, 10), font=font_sm)
        for ci, (c_tag, _, _) in enumerate(conditions):
            flat_idx = ci * len(test_cases) + ri
            fp = all_clean / f"{flat_idx:02d}_{c_tag}__{slug}.png"
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 32))
            d.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "ablation_2x2_gouache_vs_wiki.jpg", quality=92)

    # Also save a small sheet comparing Raw Cutout vs Gouache-12 vs Strong Gouache-8 inputs directly so user can see the pre-flattening strength!
    in_cols = ["Raw Photo Cutout", "Gouache-12 Pre-Flatten", "Strong Gouache-8 Pre-Flatten"]
    W_in = len(in_cols) * cw
    sheet_in = Image.new("RGB", (W_in, H), (248, 248, 248))
    din = ImageDraw.Draw(sheet_in)
    for ci, t in enumerate(in_cols):
        din.text((ci * cw + 12, 12), t, fill=(20, 20, 20), font=font)
    for ri, (_, slug, common, _, _) in enumerate(test_cases):
        y_top = 48 + ri * (ch + 32)
        din.rectangle([0, y_top, W_in, y_top + 32], fill=(232, 232, 236))
        din.text((14, y_top + 6), f"{common} ({slug}) — Input Pre-Flattening Comparison", fill=(10, 10, 10), font=font_sm)
        for ci, d_in in enumerate([raw_in, flat_in, strong_in]):
            fp = d_in / f"{ri:02d}_{slug}.png"
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet_in.paste(cell, (ci * cw, y_top + 32))
            din.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))
    sheet_in.save(ARTIFACT_DIR / "preflatten_inputs_compare.jpg", quality=92)
    print("Saved ablation_2x2_gouache_vs_wiki.jpg and preflatten_inputs_compare.jpg", flush=True)


if __name__ == "__main__":
    run_ablation()
