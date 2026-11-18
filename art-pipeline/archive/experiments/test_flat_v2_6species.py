"""Test True Luminance-Stepped Flat-Zone Input v2 across all 6 key species:
1. Mule Deer (mule_deer)
2. Muskrat (muskrat)
3. California Ground Squirrel (california_ground_squirrel)
4. American Pika (american_pika)
5. Pond Slider (pond_slider - using clean ref 120834277 with shell 3D glare suppressed & yellow neck stripes preserved)
6. Common Watersnake (common_watersnake - kept in test suite to ensure zero regression)

Columns:
- Col 1: Previous Best (Col 3 from next_steps_1ab_2_compare.jpg + v5 Watersnake)
- Col 2: Flat-Zone v2 Input + Sanitized Matte/Ink Prompt (step1000, s=0.75, mul=1.0, CFG=5.5)
- Col 3: Flat-Zone v2 Input + Sanitized Matte/Ink Prompt (step1250, s=0.78, mul=1.0, CFG=5.5)
- Col 4: Flat-Zone v2 Input + Sanitized Matte/Ink Prompt (step1250, s=0.80, mul=1.0, CFG=5.5)
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
PREV_1AB_DIR = Path(__file__).parent / "batch_run_1ab_2_test"
TEST_DIR = Path(__file__).parent / "batch_run_flat_v2_6sp"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SPECIES = [
    ("mule_deer", "Mule Deer", True),
    ("muskrat", "Muskrat", True),
    ("california_ground_squirrel", "California Ground Squirrel", True),
    ("american_pika", "American Pika", True),
    ("pond_slider", "Pond Slider", False),
    ("common_watersnake", "Common Watersnake", False),
]

NEG_INK_MATTE = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy shell, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, soft blended brushwork, soft color blending, oil painting, blurry fur, mottled fur, "
    "white fur streaks, silver hair highlights, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, extra limbs, fused legs, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_INK_MATTE = (
    "clean stylized 2d wildlife illustration, flat matte cel-shaded color zones with hard-edged stepped shading, "
    "solid uniform flat color blocks with sparse crisp dark ink linework and dark ink contour marks on top, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, zero soft gradients, "
    "isolated on a plain solid white background"
)


def sanitize_mammal_markings(text: str) -> str:
    """Replace VLM words that trigger soft painterly/blended fur in SDXL ('mottled', 'fur', 'dusky hairs', 'bushy')
    with flat matte color-zone and ink-line terminology."""
    replacements = [
        ("mottled gray-brown back", "flat gray-brown dorsal coat zone"),
        ("mottled", "two-tone flat"),
        ("whitish fur around eyes", "pale cream eye ring"),
        ("dusky hairs", "sparse dark ink line marks"),
        ("bushy tail", "stylized tail with dark ink marks"),
        ("dense fur", "flat matte coat"),
        ("densely furred feet", "clean paws"),
        ("medium brown fur", "uniform warm brown matte coat"),
        ("fur", "matte coat"),
    ]
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def make_flat_v2_mammal(src_path: Path, dst_path: Path, has_white_patch: bool = False) -> None:
    """True Luminance-Stepped Flat-Zone Input for mammals:
    - Wide 65px 25th-percentile glare suppression eliminates backlit white rims (Squirrel) and wide wet-fur triangles (Muskrat)
    - Quantizes ONLY Luminance into flat cel steps so R:G:B ratios stay locked (zero rainbow contour rings!)
    - Preserves ONLY deep anatomical darks (eyes, nostrils, mouth, paw/claw separations), erasing 100% of photographic hair grain."""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Wide 65px 25th-percentile captures the true undercoat color even across wide backlit rims or wet fur clumps
    lum_undercoat = percentile_filter(lum, percentile=25, size=65)
    max_cap = 195.0 if has_white_patch else 162.0
    clamped_lum = np.minimum(np.minimum(lum, lum_undercoat + 20.0), max_cap)

    # Also replace desaturated bright glare (where saturation is low and lum > 135) with local chromaticity
    scale = clamped_lum / np.maximum(lum, 1.0)
    suppressed = np.clip(arr * scale[:, :, None], 0, 255)

    # Median filter (size=17) erases all hair strands, followed by Gaussian smoothing into broad color zones
    med_rgb = np.stack([percentile_filter(suppressed[:, :, c], percentile=50, size=17) for c in range(3)], axis=-1)
    smooth_rgb = np.stack([gaussian_filter(med_rgb[:, :, c], sigma=5.5) for c in range(3)], axis=-1)

    # Quantize ONLY Luminance into flat discrete steps (step=24) -> ZERO rainbow contour banding!
    s_lum = np.maximum(0.299 * smooth_rgb[:, :, 0] + 0.587 * smooth_rgb[:, :, 1] + 0.114 * smooth_rgb[:, :, 2], 1.0)
    q_lum = np.round(s_lum / 24.0) * 24.0
    flat_zones = np.clip(smooth_rgb * (q_lum / s_lum)[:, :, None], 0, 255)

    # Re-inject ONLY deep anatomical darks (eyes, nostrils, mouth cleft, paw claws) — NOT coat fur grain
    lum_3 = gaussian_filter(lum, sigma=1.4)
    lum_nb = percentile_filter(lum_3, percentile=60, size=21)
    anat_drop = np.where(lum_3 < 85.0, np.clip(lum_nb - lum_3 - 14.0, 0, 255), 0.0) / 255.0
    dark_mult = np.clip(1.0 - anat_drop * 1.5, 0.08, 1.0)[:, :, None]

    out_rgb = np.where(fg[:, :, None], np.clip(flat_zones * dark_mult, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def make_flat_v2_slider(src_path: Path, dst_path: Path) -> None:
    """Suppress the 3D specular sky/sun glare on the Pond Slider's wet carapace while keeping
    the yellow neck/leg stripes and red ear marking 100% sharp!"""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / np.maximum(max_c, 1.0)

    # Sky/sun glare on the wet carapace has high bluish/gray luminance (lum > 105) and low saturation (sat < 0.28)
    # Replace glare pixels with warm matte olive-brown shell color while leaving yellow/red stripes (high sat) untouched!
    glare_weight = np.clip((lum - 100.0) / 55.0, 0.0, 1.0) * np.clip((0.32 - sat) / 0.20, 0.0, 1.0)
    warm_shell = np.stack([
        np.full_like(r, 112.0),
        np.full_like(g, 96.0),
        np.full_like(b, 68.0),
    ], axis=-1)
    deglared = arr * (1.0 - glare_weight[:, :, None]) + warm_shell * glare_weight[:, :, None]

    # Gently flatten micro-gloss via small 5px median + luminance step quantization (preserving thin yellow stripes)
    med_rgb = np.stack([percentile_filter(deglared[:, :, c], percentile=50, size=5) for c in range(3)], axis=-1)
    s_lum = np.maximum(0.299 * med_rgb[:, :, 0] + 0.587 * med_rgb[:, :, 1] + 0.114 * med_rgb[:, :, 2], 1.0)
    q_lum = np.minimum(np.round(s_lum / 18.0) * 18.0, 175.0)
    out_rgb = np.where(fg[:, :, None], np.clip(med_rgb * (q_lum / s_lum)[:, :, None], 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def make_flat_v2_watersnake(src_path: Path, dst_path: Path) -> None:
    """Keep Common Watersnake crisp while neutralizing the green leaf & blue water reflection inside the coil."""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Neutralize green leaf (g > r + 8) or blue water glare (b > r + 8) into warm dark brown/tan scale tones
    is_green_or_blue = ((g > r + 8.0) | (b > r + 8.0)) & fg
    warm_scale = np.stack([lum * 1.08, lum * 0.96, lum * 0.80], axis=-1)
    cleaned = np.where(is_green_or_blue[:, :, None], np.clip(warm_scale, 0, 170), arr)
    Image.fromarray(cleaned.astype(np.uint8)).save(dst_path)


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, lora_path: Path, strength: str, cfg: str, mul: str, n_expected: int = 6) -> None:
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
    in_dir = TEST_DIR / "inputs_v2"
    in_dir.mkdir(exist_ok=True)

    vlm_cache = json.loads((V5_EVAL / "vlm_11_cache_v2.json").read_text(encoding="utf-8"))

    # Build inputs & prompts for all 6 species
    raw_sources = {
        "mule_deer": PREV_1AB_DIR / "inputs_raw" / "00_mule_deer.png",
        "muskrat": PREV_1AB_DIR / "inputs_raw" / "01_muskrat.png",
        "california_ground_squirrel": PREV_1AB_DIR / "inputs_raw" / "02_california_ground_squirrel.png",
        "american_pika": PREV_1AB_DIR / "inputs_raw" / "03_american_pika.png",
        "pond_slider": PREV_1AB_DIR / "inputs_slider" / "01_slider_120834277.png",
        "common_watersnake": V5_EVAL / "inputs" / "09_common_watersnake.png",
    }

    prompts = []
    for i, (slug, common, is_mammal) in enumerate(SPECIES):
        dst = in_dir / f"{i:02d}_{slug}.png"
        if not dst.exists():
            if is_mammal:
                make_flat_v2_mammal(raw_sources[slug], dst, has_white_patch=(slug == "mule_deer"))
            elif slug == "pond_slider":
                make_flat_v2_slider(raw_sources[slug], dst)
            else:
                make_flat_v2_watersnake(raw_sources[slug], dst)

        if slug == "pond_slider":
            p = (
                "animl_gouache_style, a pond slider, full body wildlife illustration, single animal "
                "standing with webbed clawed feet and short tail clearly visible, "
                "flat matte olive-brown oval carapace with dark scutes bordered by pale yellow-cream rings, "
                "dark olive-green head and limbs with bright yellow longitudinal stripes and a distinct red stripe behind the eye, "
                f"{STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
            )
        else:
            res = vlm_cache[slug]
            article = "an" if common[0].lower() in "aeiou" else "a"
            missing = res.get("missing_parts_to_fill", "").strip()
            colors = sanitize_mammal_markings(res.get("visible_color_and_markings", "").strip())
            if missing:
                p = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal {missing}, {colors}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
            else:
                p = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal, {colors}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
        prompts.append(p)

    pfile = TEST_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    lora_1000 = LORA / "output" / "animl_style_v5-step00001000.safetensors"
    lora_1250 = LORA / "output" / "animl_style_v5-step00001250.safetensors"

    print("Running Col 2 (Flat-Zone v2 + step1000, s=0.75)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col2_s075_step1000", lora_1000, "0.75", "5.5", "1.0")

    print("Running Col 3 (Flat-Zone v2 + step1250, s=0.78)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col3_s078_step1250", lora_1250, "0.78", "5.5", "1.0")

    print("Running Col 4 (Flat-Zone v2 + step1250, s=0.80)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col4_s080_step1250", lora_1250, "0.80", "5.5", "1.0")

    # Assemble 6-row x 4-col comparison sheet
    CELL = 440
    HEADER_H = 55
    ROW_LABEL_H = 36
    COLS = 4
    ROWS = 6
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
        "1. Prev Col 3 Baseline",
        "2. Flat-Zone v2 (step1000, s=0.75)",
        "3. Flat-Zone v2 (step1250, s=0.78)",
        "4. Flat-Zone v2 (step1250, s=0.80)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 14, 16), title, fill=(20, 20, 30), font=font_b)

    prev_mammal_imgs = sorted((PREV_1AB_DIR / "col3_flat_mul100").glob("*.png"))
    prev_slider_img = sorted((PREV_1AB_DIR / "slider_outs").glob("*.png"))[1]
    prev_snake_img = sorted((V5_EVAL / "clean_step1000_s0.75").glob("*common_watersnake*.png"))[0]

    col2_imgs = sorted((TEST_DIR / "col2_s075_step1000").glob("*.png"))
    col3_imgs = sorted((TEST_DIR / "col3_s078_step1250").glob("*.png"))
    col4_imgs = sorted((TEST_DIR / "col4_s080_step1250").glob("*.png"))

    for r_idx, (slug, common, _) in enumerate(SPECIES):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), f"{common} ({slug})", fill=(30, 30, 50), font=font_r)

        if r_idx < 4:
            c1 = clean_post(prev_mammal_imgs[r_idx])
        elif r_idx == 4:
            c1 = clean_post(prev_slider_img)
        else:
            c1 = Image.open(prev_snake_img).convert("RGB")

        row_imgs = [
            c1,
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

    out_sheet = ARTIFACT_DIR / "flat_v2_6species_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}")


if __name__ == "__main__":
    main()
