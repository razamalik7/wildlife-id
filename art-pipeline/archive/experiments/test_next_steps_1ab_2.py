"""Test Next Steps 1a, 1b, and 2:
- Mammals (Mule Deer, Muskrat, California Ground Squirrel, American Pika):
  Col 1: v5 + Ink/Matte Prompt (CFG=5.5, s=0.78, mul=1.0, Raw Cutout) [Best from previous test]
  Col 2 (1b): v5 + Higher LoRA Multiplier (--network_mul 1.25, CFG=5.5, s=0.78, Raw Cutout)
  Col 3 (1a): v5 + Highlight-Clamped Flat-Coat Input (wide 35px highlight suppression + sharp dark ink crevices, mul=1.0, s=0.75, CFG=5.5)
  Col 4 (1a+1b): v5 + Highlight-Clamped Flat-Coat Input + Higher LoRA Multiplier (--network_mul 1.20, s=0.76, CFG=5.5)

- Pond Slider (Row 5):
  Col 1: Previous v5 (clipped back-right shell)
  Col 2: Original Photo (64270279) with ONLY top-left branch masked (back shell & hind legs 100% intact!)
  Col 3: New Curated Ref Photo #1 (120834277 - 3/4 profile with red ear stripe, full shell, front & hind legs)
  Col 4: New Funnel Ref Photo #2 (714194027 - standing on log with all 4 legs visible)
"""

import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import percentile_filter, gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
PREV_INK_TEST = Path(__file__).parent / "batch_run_mammal_ink_test"
TEST_DIR = Path(__file__).parent / "batch_run_1ab_2_test"
CACHE_DIR = Path(__file__).parent / "vetting" / "funnel_cache"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

MAMMALS = [
    (4, "mule_deer", "Mule Deer"),
    (5, "muskrat", "Muskrat"),
    (6, "california_ground_squirrel", "California Ground Squirrel"),
    (7, "american_pika", "American Pika"),
]

NEG_INK_MATTE = (
    "photo, photorealistic, 3d render, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, soft blended brushwork, oil painting, blurry fur, "
    "white fur streaks, silver hair highlights, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, extra limbs, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_INK_MATTE = (
    "clean stylized 2d wildlife illustration, flat matte cel-shaded color zones, "
    "solid uniform base coat with fine dark ink fur linework and crisp dark ink hatching marks on top, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, "
    "isolated on a plain solid white background"
)


def make_highlight_clamped_input(src_path: Path, dst_path: Path) -> None:
    """Suppress wide bright specular hair glints / wet-fur highlights into flat matte color zones
    while preserving sharp dark eyes, claws, and thin dark fur shadow lines."""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Wide 35px 30th-percentile captures the dark/mid-tone undercoat even across 25-30px wet fur clumps
    lum_undercoat = percentile_filter(lum, percentile=30, size=35)
    # Check if region is a genuinely white anatomical patch (e.g. Mule Deer throat/rump patch)
    lum_anatomical_white = percentile_filter(lum, percentile=35, size=45)

    # Clamp any pixel brighter than local undercoat down toward the undercoat (unless anatomical white patch)
    clamped_lum = np.where(
        lum_anatomical_white > 175,
        lum * 0.94,
        np.minimum(lum, lum_undercoat + 10.0),
    )
    scale = clamped_lum / np.maximum(lum, 1.0)
    suppressed = np.clip(arr * scale[:, :, None], 0, 255)

    # Smooth into broad flat matte color zones and quantize into discrete cel steps
    smooth_zones = np.stack([gaussian_filter(suppressed[:, :, c], sigma=6.0) for c in range(3)], axis=-1)
    flat_zones = np.round(smooth_zones / 16.0) * 16.0

    # Extract ONLY sharp dark features (eyes, nostrils, claws, thin dark fur shadow lines)
    lum_local_med = percentile_filter(lum, percentile=55, size=11)
    dark_lines = np.clip(lum_local_med - lum - 4.0, 0, 255) / 255.0
    dark_mult = np.clip(1.0 - dark_lines * 0.95, 0.08, 1.0)[:, :, None]

    out_rgb = np.where(fg[:, :, None], np.clip(flat_zones * dark_mult, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def prepare_pond_slider_unclipped_64270279(dst_path: Path) -> None:
    """Original Pond Slider (64270279) with ONLY the top-left branch removed so the back shell & hind legs are 100% intact."""
    raw = Image.open(CACHE_DIR / "64270279.jpg").convert("RGBA")
    cut = remove(raw)
    box = subject_bbox(cut)
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    ox, oy = (1024 - nw) // 2, (1024 - nh) // 2
    canvas.paste(subj, (ox, oy), subj)

    arr = np.array(canvas)
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    yy, xx = np.ogrid[:1024, :1024]
    # Remove ONLY the pale branch sticking out on the left (xx < 355, 300 < yy < 620)
    # Leave the back of the shell and hind legs (xx >= 355) 100% untouched!
    is_left_branch = (xx < 355) & (yy > 300) & (yy < 620)
    is_bottom_reflection = (yy > 815) & ((r + g + b) > 380)
    arr[is_left_branch | is_bottom_reflection] = [255, 255, 255, 255]
    Image.fromarray(arr).convert("RGB").save(dst_path)


def prepare_clean_cutout(photo_id: int, dst_path: Path) -> None:
    raw = Image.open(CACHE_DIR / f"{photo_id}.jpg").convert("RGBA")
    cut = remove(raw)
    box = subject_bbox(cut)
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    ox, oy = (1024 - nw) // 2, (1024 - nh) // 2
    canvas.paste(subj, (ox, oy), subj)
    canvas.convert("RGB").save(dst_path)


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, lora_path: Path, strength: str, cfg: str, mul: str, n_expected: int = 4) -> None:
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= n_expected:
        print(f"  [cached] {out_dir.name}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
        "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        "--outdir", str(out_dir),
        "--network_module", "networks.lora",
        "--network_weights", str(lora_path),
        "--network_mul", mul,
        "--max_embeddings_multiples", "3",
        "--from_file", str(prompt_file),
        "--image_path", str(in_dir),
        "--use_original_file_name",
        "--strength", strength,
        "--W", "1024", "--H", "1024",
        "--steps", "28", "--sampler", "euler_a",
        "--scale", cfg, "--bf16", "--sdpa",
        "--batch_size", "1", "--images_per_prompt", "1",
    ]
    subprocess.run(cmd, check=True)


def clean_post(img_path: Path) -> Image.Image:
    im = Image.open(img_path).convert("RGBA")
    cut = remove(im)
    box = subject_bbox(cut)
    subj = cut.crop(box)
    scale = int(1024 * 0.78) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    return canvas.convert("RGB")


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    raw_in_dir = TEST_DIR / "inputs_raw"
    flat_in_dir = TEST_DIR / "inputs_flat"
    slider_in_dir = TEST_DIR / "inputs_slider"
    raw_in_dir.mkdir(exist_ok=True)
    flat_in_dir.mkdir(exist_ok=True)
    slider_in_dir.mkdir(exist_ok=True)

    vlm_cache = json.loads((V5_EVAL / "vlm_11_cache_v2.json").read_text(encoding="utf-8"))

    prompts_mammal = []
    for i, (orig_idx, slug, common) in enumerate(MAMMALS):
        src_inp = V5_EVAL / "inputs" / f"{orig_idx:02d}_{slug}.png"
        dst_raw = raw_in_dir / f"{i:02d}_{slug}.png"
        dst_flat = flat_in_dir / f"{i:02d}_{slug}.png"
        if not dst_raw.exists():
            Image.open(src_inp).convert("RGB").save(dst_raw)
        if not dst_flat.exists():
            make_highlight_clamped_input(dst_raw, dst_flat)

        res = vlm_cache[slug]
        article = "an" if common[0].lower() in "aeiou" else "a"
        missing = res.get("missing_parts_to_fill", "").strip()
        colors = res.get("visible_color_and_markings", "").strip()
        colors_clean = colors.replace("dense fur", "flat matte coat").replace("medium brown fur", "medium brown coat")
        if missing:
            p = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal {missing}, {colors_clean}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
        else:
            p = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal, {colors_clean}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
        prompts_mammal.append(p)

    pfile_mammal = TEST_DIR / "prompts_mammal.txt"
    pfile_mammal.write_text("\n".join(prompts_mammal), encoding="utf-8")

    # Prepare 3 Pond Slider inputs:
    # 00: Original 64270279 with ONLY left branch removed (back shell intact)
    # 01: Curated 120834277 (3/4 profile with red ear stripe, full shell, front & hind legs)
    # 02: Funnel 714194027 (standing on log, 4 legs visible)
    if not (slider_in_dir / "00_slider_unclipped.png").exists():
        prepare_pond_slider_unclipped_64270279(slider_in_dir / "00_slider_unclipped.png")
    if not (slider_in_dir / "01_slider_120834277.png").exists():
        prepare_clean_cutout(120834277, slider_in_dir / "01_slider_120834277.png")
    if not (slider_in_dir / "02_slider_714194027.png").exists():
        prepare_clean_cutout(714194027, slider_in_dir / "02_slider_714194027.png")

    slider_prompt = (
        "animl_gouache_style, a pond slider, full body wildlife illustration, single animal "
        "standing with all four webbed clawed feet and short tail clearly visible, "
        "olive-brown oval carapace with dark scutes bordered by pale yellow-cream rings, "
        "dark olive-green head and limbs with bright yellow longitudinal stripes and a distinct red stripe behind the eye, "
        f"{STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
    )
    pfile_slider = TEST_DIR / "prompts_slider.txt"
    pfile_slider.write_text("\n".join([slider_prompt] * 3), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v5-step00001250.safetensors"
    lora_1000 = LORA / "output" / "animl_style_v5-step00001000.safetensors"

    # Run mammal tests (Col 2 = 1b mul 1.25 on raw; Col 3 = 1a flat input mul 1.0; Col 4 = 1a+1b flat input mul 1.20)
    print("Running Col 2 (1b: Raw Cutout + LoRA mul=1.25)...")
    run_sdxl(raw_in_dir, pfile_mammal, TEST_DIR / "col2_mul125", lora_1250, "0.78", "5.5", "1.25")

    print("Running Col 3 (1a: Highlight-Clamped Input + LoRA mul=1.0)...")
    run_sdxl(flat_in_dir, pfile_mammal, TEST_DIR / "col3_flat_mul100", lora_1000, "0.75", "5.5", "1.0")

    print("Running Col 4 (1a+1b: Highlight-Clamped Input + LoRA mul=1.20)...")
    run_sdxl(flat_in_dir, pfile_mammal, TEST_DIR / "col4_flat_mul120", lora_1250, "0.76", "5.5", "1.20")

    print("Running Pond Slider 3 candidates...")
    out_slider = TEST_DIR / "slider_outs"
    run_sdxl(slider_in_dir, pfile_slider, out_slider, lora_1000, "0.75", "5.5", "1.0", n_expected=3)

    # Assemble 5-row x 4-col comparison sheet
    CELL = 460
    HEADER_H = 55
    ROW_LABEL_H = 38
    COLS = 4
    ROWS = 5
    W = COLS * CELL
    H = HEADER_H + ROWS * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 18)
        font_r = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. Prev Best (v5 Ink Prompt, mul=1.0)",
        "2. [1b] Higher LoRA Weight (mul=1.25)",
        "3. [1a] Highlight-Clamped Input (mul=1.0)",
        "4. [1a+1b] Clamped Input + mul=1.20",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 14, 16), title, fill=(20, 20, 30), font=font_b)

    prev_col3_imgs = sorted((PREV_INK_TEST / "clean_col3_v5_unetonly_s078").glob("*.png"))
    col2_imgs = sorted((TEST_DIR / "col2_mul125").glob("*.png"))
    col3_imgs = sorted((TEST_DIR / "col3_flat_mul100").glob("*.png"))
    col4_imgs = sorted((TEST_DIR / "col4_flat_mul120").glob("*.png"))

    for r_idx, (orig_idx, slug, common) in enumerate(MAMMALS):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 9), f"{common} ({slug})", fill=(30, 30, 50), font=font_r)

        row_imgs = [
            Image.open(prev_col3_imgs[r_idx]).convert("RGB"),
            clean_post(col2_imgs[r_idx]),
            clean_post(col3_imgs[r_idx]),
            clean_post(col4_imgs[r_idx]),
        ]
        for c_idx, im in enumerate(row_imgs):
            thumb = im.resize((CELL - 8, CELL - 8), Image.LANCZOS)
            sheet.paste(thumb, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle(
                [c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL],
                outline=(210, 215, 220), width=1
            )

    # Row 5: Pond Slider comparison
    y0 = HEADER_H + 4 * (CELL + ROW_LABEL_H)
    draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(220, 238, 228))
    slider_labels = [
        "Pond Slider: 1. Prev Clipped Shell",
        "2. Unclipped Back Shell (64270279)",
        "3. New Curated Ref (120834277)",
        "4. New Standing Ref (714194027)",
    ]
    for c_idx, lbl in enumerate(slider_labels):
        draw.text((c_idx * CELL + 14, y0 + 9), lbl, fill=(20, 50, 30), font=font_r)

    prev_slider = Image.open(sorted((V5_EVAL / "clean_step1000_s0.75").glob("*pond_slider*.png"))[0]).convert("RGB")
    s_outs = [clean_post(p) for p in sorted(out_slider.glob("*.png"))]
    slider_row = [prev_slider] + s_outs
    for c_idx, im in enumerate(slider_row):
        thumb = im.resize((CELL - 8, CELL - 8), Image.LANCZOS)
        sheet.paste(thumb, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
        draw.rectangle(
            [c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL],
            outline=(210, 215, 220), width=1
        )

    out_sheet = ARTIFACT_DIR / "next_steps_1ab_2_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}")


if __name__ == "__main__":
    main()
