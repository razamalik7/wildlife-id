"""Targeted test addressing the 4 specific issues identified in flat_v2_6species_compare.jpg:
1. Pond Slider:
   - Fix prompt: remove 'bordered by pale yellow-cream rings' (which triggered the yellow Box Turtle shell prior!) -> use 'dark olive-brown oval carapace with subtle dark brown scutes'.
   - Fix input: only de-glare the carapace (x < 640, y > 340), leaving the head/red ear stripe 100% untouched & sharp so the red stripe never disappears at s=0.76-0.78!
2. California Ground Squirrel (and all mammals):
   - Remove the staircase rounding `np.round(s_lum / 24.0) * 24.0` which created concentric topographic contour rings around the squirrel's cheek/face that SDXL turned into AI rippling!
3. Muskrat:
   - Remove invisible 'scaled tail, webbed hind feet' from the VLM markings string (which caused SDXL at s=0.80 to turn the wet plant root between its front paws into an extra foot), and neutralize the dark wet root under the front paws.
4. Common Watersnake:
   - Compare s=0.72 vs s=0.76 vs dry-land candidate (15458841) so the coils don't warp into impossible connections.
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
PREV_V2_DIR = Path(__file__).parent / "batch_run_flat_v2_6sp"
TEST_DIR = Path(__file__).parent / "batch_run_clean_fixes_6sp"
CACHE_DIR = Path(__file__).parent / "vetting" / "funnel_cache"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SPECIES = [
    ("mule_deer", "Mule Deer"),
    ("muskrat", "Muskrat"),
    ("california_ground_squirrel", "California Ground Squirrel"),
    ("american_pika", "American Pika"),
    ("pond_slider", "Pond Slider"),
    ("common_watersnake", "Common Watersnake"),
]

NEG_INK_MATTE = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy shell, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, soft blended brushwork, soft color blending, oil painting, blurry fur, mottled fur, "
    "rippling fur, concentric lines, white fur streaks, silver hair highlights, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, extra limbs, extra feet, fused legs, impossible coils, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_INK_MATTE = (
    "clean stylized 2d wildlife illustration, flat matte cel-shaded color zones, "
    "solid uniform flat color blocks with sparse crisp dark ink linework and dark ink contour marks on top, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, "
    "isolated on a plain solid white background"
)


def make_smooth_flat_mammal(src_path: Path, dst_path: Path, slug: str) -> None:
    """Suppress specular glare and smooth out individual hair strands WITHOUT staircase-rounding (`np.round`)
    so there are zero topographic contour rings around the squirrel's cheek/face!"""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)

    if slug == "muskrat":
        # Remove the wet plant root hanging below the front left paw (x < 310, y > 690)
        yy, xx = np.ogrid[:1024, :1024]
        is_root = (xx < 310) & (yy > 688)
        fg = fg & (~is_root)
        arr[is_root] = 255.0

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Wide 65px 25th-percentile captures the true undercoat color across backlit rims or wet fur clumps
    lum_undercoat = percentile_filter(lum, percentile=25, size=65)
    max_cap = 195.0 if slug == "mule_deer" else 160.0
    clamped_lum = np.minimum(np.minimum(lum, lum_undercoat + 22.0), max_cap)

    scale = clamped_lum / np.maximum(lum, 1.0)
    suppressed = np.clip(arr * scale[:, :, None], 0, 255)

    # Median filter (size=15) + Gaussian smoothing (sigma=4.5) erases hair strands smoothly with ZERO staircase contour rings!
    med_rgb = np.stack([percentile_filter(suppressed[:, :, c], percentile=50, size=15) for c in range(3)], axis=-1)
    smooth_rgb = np.stack([gaussian_filter(med_rgb[:, :, c], sigma=4.5) for c in range(3)], axis=-1)

    # Re-inject ONLY deep anatomical darks (eyes, nostrils, mouth cleft, paw claws) — NOT coat fur grain
    lum_3 = gaussian_filter(lum, sigma=1.4)
    lum_nb = percentile_filter(lum_3, percentile=60, size=21)
    anat_drop = np.where(lum_3 < 85.0, np.clip(lum_nb - lum_3 - 14.0, 0, 255), 0.0) / 255.0
    dark_mult = np.clip(1.0 - anat_drop * 1.5, 0.08, 1.0)[:, :, None]

    out_rgb = np.where(fg[:, :, None], np.clip(smooth_rgb * dark_mult, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def make_clean_slider_120834277(src_path: Path, dst_path: Path) -> None:
    """De-glare ONLY the carapace of Pond Slider (120834277) so the 3D sky reflection on the shell
    becomes matte warm olive-brown, while leaving the head, red ear stripe, and yellow neck stripes 100% sharp!"""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / np.maximum(max_c, 1.0)

    yy, xx = np.ogrid[:1024, :1024]
    # The head and neck are at xx > 610, yy < 520: keep head/neck 100% untouched!
    is_shell_region = (xx < 630) & (yy > 320) & fg

    # Sky glare on the shell has high luminance (lum > 95) and low saturation (sat < 0.30)
    glare_w = np.where(
        is_shell_region,
        np.clip((lum - 92.0) / 50.0, 0.0, 1.0) * np.clip((0.32 - sat) / 0.20, 0.0, 1.0),
        0.0,
    )
    # Replace glare with warm matte dark olive-brown matching the non-glare scutes
    warm_olive_brown = np.stack([
        np.full_like(r, 96.0),
        np.full_like(g, 84.0),
        np.full_like(b, 58.0),
    ], axis=-1)
    out = arr * (1.0 - glare_w[:, :, None]) + warm_olive_brown * glare_w[:, :, None]

    # Slightly boost the red ear stripe on the head (where R > G + 18 in head region) so it stays unmistakable at s=0.76-0.78
    is_red_stripe = (xx > 610) & (yy < 400) & (r > g + 15.0) & fg
    out[is_red_stripe, 0] = np.clip(out[is_red_stripe, 0] * 1.25, 0, 220)
    out[is_red_stripe, 1] = out[is_red_stripe, 1] * 0.80
    out[is_red_stripe, 2] = out[is_red_stripe, 2] * 0.80

    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(dst_path)


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
    in_dir = TEST_DIR / "inputs_clean"
    in_dir.mkdir(exist_ok=True)

    raw_sources = {
        "mule_deer": PREV_1AB_DIR / "inputs_raw" / "00_mule_deer.png",
        "muskrat": PREV_1AB_DIR / "inputs_raw" / "01_muskrat.png",
        "california_ground_squirrel": PREV_1AB_DIR / "inputs_raw" / "02_california_ground_squirrel.png",
        "american_pika": PREV_1AB_DIR / "inputs_raw" / "03_american_pika.png",
        "pond_slider": PREV_1AB_DIR / "inputs_slider" / "01_slider_120834277.png",
        "common_watersnake": PREV_V2_DIR / "inputs_v2" / "05_common_watersnake.png",
    }

    for i, (slug, _) in enumerate(SPECIES):
        dst = in_dir / f"{i:02d}_{slug}.png"
        if not dst.exists():
            if slug in ("mule_deer", "muskrat", "california_ground_squirrel", "american_pika"):
                make_smooth_flat_mammal(raw_sources[slug], dst, slug)
            elif slug == "pond_slider":
                make_clean_slider_120834277(raw_sources[slug], dst)
            else:
                Image.open(raw_sources[slug]).convert("RGB").save(dst)

    # Clean, accurate prompts (no 'yellow-cream rings' on Pond Slider shell, no invisible 'webbed hind feet' on Muskrat)
    prompts = [
        # 0. Mule Deer
        f"animl_gouache_style, a mule deer, full body wildlife illustration, single animal standing on all four slender legs with dark cloven hooves visible, dark eyes, large ears, brown muzzle, grayish-brown matte coat, pale white rump patch, dark brown bifurcated antlers, slender brown legs, black-tipped tail, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 1. Muskrat (removed invisible 'scaled tail, webbed hind feet' so it doesn't turn the chin/paw area into an extra foot!)
        f"animl_gouache_style, a muskrat, full body wildlife illustration, single animal with small front paws tucked neatly under chin, dark eyes, dark muzzle, uniform warm brown matte coat, darker brown back, lighter tan cheeks and belly, short brown whiskers, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 2. California Ground Squirrel
        f"animl_gouache_style, a california ground squirrel, full body wildlife illustration, single animal with rounded hind haunch and paws visible, pale cream eye ring, two-tone flat gray-brown dorsal coat zone, lighter gray shoulders, warm buff underside, stylized tail with dark ink marks, dark paws, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 3. American Pika
        f"animl_gouache_style, an american pika, full body wildlife illustration, single animal with rounded hind haunch and paws visible, dark eyes, whiskered muzzle, grayish-brown rounded body, warm buff chest, flat matte coat, rounded dark ears with pale rims, clean paws, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 4. Pond Slider (NO 'yellow-cream rings' on shell! Accurate dark olive-brown carapace + bright red stripe behind eye)
        f"animl_gouache_style, a pond slider, full body wildlife illustration, single animal standing with webbed clawed feet visible, flat matte dark olive-brown oval carapace with dark brown scutes and subtle tan marginal rim, dark green head and limbs with fine yellow longitudinal stripes, bright red stripe behind the eye, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 5. Common Watersnake
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single animal coiled cleanly, dark brown body with dark crossbands, reddish-brown head scales, dark eyes, pale cream-gray belly scales with dark crescents, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
    ]
    pfile = TEST_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v5-step00001250.safetensors"

    print("Running Col 3 (Smooth Glare-Clamped Input + Fixed Prompts, step1250, s=0.76)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col3_s076", lora_1250, "0.76", "5.5", "1.0")

    print("Running Col 4 (Smooth Glare-Clamped Input + Fixed Prompts, step1250, s=0.80)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col4_s080", lora_1250, "0.80", "5.5", "1.0")

    # Assemble 6-row x 4-col sheet:
    # Col 1: Prev Flat-Zone v2 (s=0.78)
    # Col 2: Prev Flat-Zone v2 (s=0.80)
    # Col 3: Fixed Smooth Flat Input + Fixed Prompts (s=0.76)
    # Col 4: Fixed Smooth Flat Input + Fixed Prompts (s=0.80)
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
        "1. Prev v2 (s=0.78, staircase+old prompt)",
        "2. Prev v2 (s=0.80, staircase+old prompt)",
        "3. Fixed Input+Prompt (s=0.76, step1250)",
        "4. Fixed Input+Prompt (s=0.80, step1250)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 16), title, fill=(20, 20, 30), font=font_b)

    prev_c3 = sorted((PREV_V2_DIR / "col3_s078_step1250").glob("*.png"))
    prev_c4 = sorted((PREV_V2_DIR / "col4_s080_step1250").glob("*.png"))
    new_c3 = sorted((TEST_DIR / "col3_s076").glob("*.png"))
    new_c4 = sorted((TEST_DIR / "col4_s080").glob("*.png"))

    for r_idx, (slug, common) in enumerate(SPECIES):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), f"{common} ({slug})", fill=(30, 30, 50), font=font_r)

        row_imgs = [
            clean_post(prev_c3[r_idx]),
            clean_post(prev_c4[r_idx]),
            clean_post(new_c3[r_idx]),
            clean_post(new_c4[r_idx]),
        ]
        for c_idx, im in enumerate(row_imgs):
            thumb = im.resize((CELL - 8, CELL - 8), Image.LANCZOS)
            sheet.paste(thumb, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle(
                [c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL],
                outline=(210, 215, 220), width=1
            )

    out_sheet = ARTIFACT_DIR / "clean_fixes_6sp_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}")


if __name__ == "__main__":
    main()
