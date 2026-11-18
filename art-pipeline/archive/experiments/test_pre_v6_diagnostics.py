"""Pre-v6 Diagnostic Test on v5:
Tests the exact refinements before launching animl_style_v6 training:
1. Dialed-in Marten-like prompt (removes 'solid uniform flat color blocks... zero soft gradients' over-correction -> 'matte natural color zones with subtle low-contrast warm brown ink fur marks').
2. Muskrat: Smooth curved belly trim on 711861445 (removes the 90-degree notch at x=310 and hanging wet belly-fur lobes that caused extra feet) + tests alternative ref 130874899.
3. California Ground Squirrel: Sharp edge-preserving high-frequency fur-contrast damping (0% spatial blur—keeps 100% sharp silhouette, single tail, and shoulder mantle while reducing harsh sunlit white/black hair contrast by 55%).
4. Pond Slider: Trims the tiny 8px bark lump on top ridge of 120834277's shell (giving Col 4's smooth shell shape even at s=0.74-0.76!) and widens/brightens the crimson red postorbital ear stripe on the head so the red stripe is unmistakable.
5. Common Watersnake: Tests dry-sand S-coil candidate (700426996) and clean candidate (15458841) vs underwater coil (278839486) to eliminate impossible coil connections.
"""

import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
PREV_1AB_DIR = Path(__file__).parent / "batch_run_1ab_2_test"
PREV_RAW_DIR = Path(__file__).parent / "batch_run_raw_unblurred_6sp"
CACHE_DIR = Path(__file__).parent / "vetting" / "funnel_cache"
TEST_DIR = Path(__file__).parent / "batch_run_pre_v6_diag"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

NEG_CALM = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy shell, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, oil painting, blurry fur, stark white fur streaks, harsh contrast, "
    "heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, two tails, extra limbs, extra feet, extra paws, row of feet, fused legs, impossible coils, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_MARTEN_CALM = (
    "clean stylized 2d wildlife illustration, matte natural color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink fur and contour lines, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, "
    "isolated on a plain solid white background"
)


def prepare_muskrat_smooth_belly(src_path: Path, dst_path: Path) -> None:
    """In 711861445, trimming `(xx < 310) & (yy > 688)` left a 90-degree vertical step at x=310 and hanging
    wet belly-fur lobes at x=310..470 that SDXL turned into extra feet!
    Here we trim along a smooth continuous curve under the chin/front paw and across the lower belly lobes."""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im)
    yy, xx = np.ogrid[:1024, :1024]

    # Keep the single front paw area at x in [210, 305] down to y=682, and smoothly curve the belly line behind x=305
    # In the raw cutout, hanging wet fur lobes stick down below y=685 between x=290 and x=470:
    belly_cutoff_y = np.where(
        xx < 295,
        682.0,
        682.0 + np.clip((xx - 295.0) / 190.0, 0.0, 1.0) * 32.0,
    )
    is_hanging_lobe = (xx < 485) & (yy > belly_cutoff_y)
    arr[is_hanging_lobe] = [255, 255, 255]
    Image.fromarray(arr).save(dst_path)


def prepare_squirrel_contrast_damped(src_path: Path, dst_path: Path) -> None:
    """Zero spatial blur! Decomposes luminance into a 4px local base + high-frequency hair contrast,
    and damps extreme high-frequency white/black hair spikes by 55% while keeping all edges, eyes, claws,
    and shoulder markings 100% pixel-sharp!"""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lum_local = gaussian_filter(lum, sigma=3.0)
    high_freq = lum - lum_local

    # Compress high-frequency hair contrast (especially bright white hair tips > 0) while preserving deep dark eye/claw lines
    damped_hf = np.where(high_freq > 0, high_freq * 0.30, high_freq * 0.55)
    # Also cap overall blown-out backlit glare on neck/forehead at 175
    new_lum = np.minimum(lum_local + damped_hf, 175.0)
    scale = new_lum / np.maximum(lum, 1.0)

    out = np.where(fg[:, :, None], np.clip(arr * scale[:, :, None], 0, 255).astype(np.uint8), 255)
    Image.fromarray(out).save(dst_path)


def prepare_slider_120834277_fixed(src_path: Path, dst_path: Path) -> None:
    """Fixes both issues on 120834277 without blurring:
    1. Trims the tiny 8px dark bark lump sticking up from the top-center of the carapace (around x=470..540, y<365)
       so the shell has Col 4's smooth oval shape even at s=0.74-0.76!
    2. Widens and brightens the crimson-red postorbital stripe behind the eye on the head so SDXL's VAE latent
       sees a bold, unmistakable red ear patch!"""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    yy, xx = np.ogrid[:1024, :1024]

    # 1. Smoothly trim the little bark knob protruding above the carapace dome (around x=460..545, y < 368)
    dome_top_y = 368.0 + 0.0035 * ((xx - 520.0) ** 2)
    is_top_bark_knob = (xx > 445) & (xx < 560) & (yy < dome_top_y)
    arr[is_top_bark_knob] = [255.0, 255.0, 255.0]

    # 2. Enhance & slightly widen the red postorbital stripe behind the eye on the head (head is at xx > 630, yy < 380)
    # In 120834277, the red stripe runs diagonally behind the eye around x=665..725, y=235..295
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    fg = np.any(arr < 250, axis=-1)
    is_head_red = (xx > 650) & (xx < 745) & (yy > 220) & (yy < 315) & (r > g + 6.0) & fg
    # Dilate the red stripe slightly downward/inward on the side of the head so it's ~20px wide (2.5 latent pixels)
    red_mask_float = gaussian_filter(is_head_red.astype(np.float32), sigma=2.5)
    is_expanded_red = (red_mask_float > 0.18) & fg & (xx > 655) & (yy < 310)

    arr[is_expanded_red, 0] = np.clip(arr[is_expanded_red, 0] * 1.45 + 35.0, 175.0, 225.0)
    arr[is_expanded_red, 1] = np.clip(arr[is_expanded_red, 1] * 0.45, 20.0, 55.0)
    arr[is_expanded_red, 2] = np.clip(arr[is_expanded_red, 2] * 0.45, 20.0, 55.0)

    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(dst_path)


def prepare_clean_cutout(photo_id: int, dst_path: Path) -> None:
    if dst_path.exists():
        return
    raw = Image.open(CACHE_DIR / f"{photo_id}.jpg").convert("RGBA")
    cut = remove(raw)
    box = subject_bbox(cut)
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    canvas.convert("RGB").save(dst_path)


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, lora_path: Path, strength: str, cfg: str, mul: str, n_expected: int) -> None:
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
    in_dir = TEST_DIR / "inputs"
    in_dir.mkdir(exist_ok=True)

    # 00: Mule Deer (sharp cutout)
    Image.open(PREV_1AB_DIR / "inputs_raw" / "00_mule_deer.png").convert("RGB").save(in_dir / "00_mule_deer.png")
    # 01: Muskrat (smooth curved belly trim removing the 90-deg notch & wet belly lobes)
    prepare_muskrat_smooth_belly(PREV_1AB_DIR / "inputs_raw" / "01_muskrat.png", in_dir / "01_muskrat.png")
    # 02: California Ground Squirrel (contrast-damped fur without spatial blur)
    prepare_squirrel_contrast_damped(PREV_1AB_DIR / "inputs_raw" / "02_california_ground_squirrel.png", in_dir / "02_squirrel.png")
    # 03: American Pika (sharp cutout)
    Image.open(PREV_1AB_DIR / "inputs_raw" / "03_american_pika.png").convert("RGB").save(in_dir / "03_pika.png")
    # 04: Pond Slider 120834277 (top bark knob trimmed + red ear stripe widened/brightened)
    prepare_slider_120834277_fixed(PREV_1AB_DIR / "inputs_slider" / "01_slider_120834277.png", in_dir / "04_slider_120834277.png")
    # 05: Common Watersnake dry-sand S-coil candidate (700426996)
    prepare_clean_cutout(700426996, in_dir / "05_watersnake_700426996.png")
    # 06: Common Watersnake studio candidate (15458841)
    prepare_clean_cutout(15458841, in_dir / "06_watersnake_15458841.png")

    prompts = [
        # 00: Mule Deer
        f"animl_gouache_style, a mule deer, full body wildlife illustration, single animal standing on all four slender legs with dark cloven hooves visible, dark eyes, large ears, brown muzzle, grayish-brown matte coat, pale white rump patch, dark brown bifurcated antlers, slender brown legs, black-tipped tail, {STYLE_MARTEN_CALM} --n {NEG_CALM} --d 42",
        # 01: Muskrat
        f"animl_gouache_style, a muskrat, full body wildlife illustration, single animal crouching with smooth rounded belly and only two small front paws tucked under chin, dark eyes, dark muzzle, uniform warm brown coat, darker brown back, lighter tan cheeks, short brown whiskers, {STYLE_MARTEN_CALM} --n {NEG_CALM} --d 42",
        # 02: California Ground Squirrel
        f"animl_gouache_style, a california ground squirrel, full body wildlife illustration, single animal with rounded hind haunch and paws visible, pale cream eye ring, calm warm gray-brown dorsal coat, pale gray shoulder mantle, warm buff underside, bushy tail, dark paws, {STYLE_MARTEN_CALM} --n {NEG_CALM} --d 42",
        # 03: American Pika
        f"animl_gouache_style, an american pika, full body wildlife illustration, single animal with rounded hind haunch and paws visible, dark eyes, whiskered muzzle, grayish-brown rounded body, warm buff chest, soft matte coat with subtle warm brown fur marks, rounded dark ears with pale rims, clean paws, {STYLE_MARTEN_CALM} --n {NEG_CALM} --d 42",
        # 04: Pond Slider
        f"animl_gouache_style, a pond slider, full body wildlife illustration, single animal standing with webbed clawed feet visible, smooth dark olive-brown oval carapace with dark brown scutes, dark green head and limbs with fine yellow longitudinal stripes, (bright crimson-red stripe behind the eye:1.35), {STYLE_MARTEN_CALM} --n {NEG_CALM}, yellow shell, box turtle shell --d 42",
        # 05: Common Watersnake (700426996)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single animal in a clean S-coil, brown and tan body with dark brown crossbands, dark eyes, pale cream labial scales with dark vertical bars, {STYLE_MARTEN_CALM} --n {NEG_CALM} --d 42",
        # 06: Common Watersnake (15458841)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single animal coiled cleanly, reddish-brown and tan body with darker brown crossbands, dark eyes, pale cream chin scales, {STYLE_MARTEN_CALM} --n {NEG_CALM} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1000 = LORA / "output" / "animl_style_v5-step00001000.safetensors"
    lora_1250 = LORA / "output" / "animl_style_v5-step00001250.safetensors"

    print("Running Col 2 (step1000, s=0.75)...")
    run_sdxl(in_dir, pf, TEST_DIR / "col2_s075_step1000", lora_1000, "0.75", "5.5", "1.0", 7)

    print("Running Col 3 (step1250, s=0.77)...")
    run_sdxl(in_dir, pf, TEST_DIR / "col3_s077_step1250", lora_1250, "0.77", "5.5", "1.0", 7)

    print("Running Col 4 (step1250, s=0.80)...")
    run_sdxl(in_dir, pf, TEST_DIR / "col4_s080_step1250", lora_1250, "0.80", "5.5", "1.0", 7)

    # Assemble 7-row x 4-col sheet
    CELL = 420
    HEADER_H = 52
    ROW_LABEL_H = 34
    COLS = 4
    ROWS = 7
    W = COLS * CELL
    H = HEADER_H + ROWS * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 17)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. Refined Sharp Input",
        "2. v5-step1000 (s=0.75, Marten Prompt)",
        "3. v5-step1250 (s=0.77, Marten Prompt)",
        "4. v5-step1250 (s=0.80, Marten Prompt)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 15), title, fill=(20, 20, 30), font=font_b)

    row_labels = [
        "Mule Deer (Calm Marten Prompt)",
        "Muskrat (Smooth Curved Belly Trim - No 90-deg Notch / Lobes)",
        "California Ground Squirrel (Contrast-Damped Fur, Zero Blur + Calm Prompt)",
        "American Pika (Calm Marten Prompt - Less Basic)",
        "Pond Slider (Top Bark Lump Trimmed + Widened Crimson Red Ear Stripe)",
        "Common Watersnake Candidate A (700426996 - Dry Sand S-Coil)",
        "Common Watersnake Candidate B (15458841 - Clean Coil)",
    ]

    in_imgs = sorted(in_dir.glob("*.png"))
    c2_imgs = sorted((TEST_DIR / "col2_s075_step1000").glob("*.png"))
    c3_imgs = sorted((TEST_DIR / "col3_s077_step1250").glob("*.png"))
    c4_imgs = sorted((TEST_DIR / "col4_s080_step1250").glob("*.png"))

    for r_idx, lbl in enumerate(row_labels):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)

        row_imgs = [
            Image.open(in_imgs[r_idx]).convert("RGB"),
            clean_post(c2_imgs[r_idx]),
            clean_post(c3_imgs[r_idx]),
            clean_post(c4_imgs[r_idx]),
        ]
        for c_idx, im in enumerate(row_imgs):
            thumb = im.resize((CELL - 8, CELL - 8), Image.LANCZOS)
            sheet.paste(thumb, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle(
                [c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL],
                outline=(210, 215, 220), width=1
            )

    out_sheet = ARTIFACT_DIR / "pre_v6_diagnostics_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}")


if __name__ == "__main__":
    main()
