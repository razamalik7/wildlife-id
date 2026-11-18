"""Complete Evaluation Suite for animl_style_v6 on the 6 Key Diagnostic Species + All 11 Unseen Species.

1. Prepares clean, non-blurred inputs for all 11 unseen species (including our perfected inputs for
   Mule Deer, Muskrat, California Ground Squirrel, American Pika, Pond Slider 120834277, and Common Watersnake 15458841).
2. Evaluates animl_style_v6 checkpoints (step1000, step1250, step1500) at calibrated denoising strengths
   with 100% training-inference caption parity (STYLE_CLAUSE matching build_dataset.py).
3. Renders high-resolution side-by-side comparison sheets against v5 into the artifact directory.
"""

import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, percentile_filter
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
PREV_1AB_DIR = Path(__file__).parent / "batch_run_1ab_2_test"
RAW_SHARP_DIR = Path(__file__).parent / "batch_run_raw_unblurred_6sp" / "inputs_sharp"
PRE_V6_DIR = Path(__file__).parent / "batch_run_pre_v6_diag"
MS_QUICK_DIR = Path(__file__).parent / "batch_run_ms_quick"
EVAL_DIR = Path(__file__).parent / "batch_run_v6_eval"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

# Exact 100% parity with animl_style_v6 training STYLE_CLAUSE in build_dataset.py
STYLE_V6 = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink fur and contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_V6 = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy shell, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, oil painting, blurry fur, stark white fur streaks, harsh contrast, "
    "heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, two tails, extra limbs, extra feet, extra paws, row of feet, claws on belly, fused legs, impossible coils, tiling, text, watermark, border, cropped body, cut off"
)


def make_clean_muskrat(dst_path: Path) -> None:
    """Clean Muskrat input that eliminated the extra middle foot in ms_quick_fix_compare.jpg,
    plus gentle top-edge smoothing on the chest arch so no harsh cutout line shows."""
    src_path = PREV_1AB_DIR / "inputs_raw" / "01_muskrat.png"
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)
    yy, xx = np.ogrid[:1024, :1024]

    smooth_belly_y = np.where(
        xx < 265,
        688.0,
        688.0 - 24.0 * np.sin(np.clip((xx - 265.0) / 240.0, 0.0, 1.0) * np.pi),
    )
    cut_mask = (xx < 505) & (yy > smooth_belly_y)
    fg = fg & (~cut_mask)
    arr[cut_mask] = 255.0

    med_flank = np.stack([percentile_filter(arr[:, :, c], percentile=45, size=15) for c in range(3)], axis=-1)
    smooth_flank = np.stack([gaussian_filter(med_flank[:, :, c], sigma=4.0) for c in range(3)], axis=-1)

    w_x = np.clip((xx - 255.0) / 45.0, 0.0, 1.0)
    w_y = np.clip((yy - 540.0) / 55.0, 0.0, 1.0)
    w_flank = (w_x * w_y)[:, :, None]

    out = arr * (1.0 - 0.85 * w_flank) + smooth_flank * (0.85 * w_flank)
    out_rgb = np.where(fg[:, :, None], np.clip(out, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def make_clean_squirrel(dst_path: Path) -> None:
    """Zero-blur boundary-preserving fur-contrast damping on California Ground Squirrel:
    preserves 100% sharp outer edges, single tail, eyes, ears, and shoulder mantle while calming both
    fine hair spikes (sigma=3) and medium diagonal flank waves (sigma=32) on the back, plus lightening
    the dark chin-shadow pocket above the paws so no berry/nut appears."""
    src_path = RAW_SHARP_DIR / "02_california_ground_squirrel.png"
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)
    yy, xx = np.ogrid[:1024, :1024]

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lum_local = gaussian_filter(lum, sigma=3.0)
    lum_broad = gaussian_filter(lum, sigma=32.0)
    high_freq = lum - lum_local

    back_w = np.clip((xx - 455.0) / 50.0, 0.0, 1.0) * np.clip((710.0 - yy) / 50.0, 0.0, 1.0) * fg
    base_lum = lum_local * (1.0 - 0.70 * back_w) + lum_broad * (0.70 * back_w)

    damped_hf = np.where(high_freq > 0, high_freq * 0.25, high_freq * 0.48)
    new_lum = np.minimum(base_lum + damped_hf, 172.0)
    scale = new_lum / np.maximum(lum, 1.0)
    out = np.clip(arr * scale[:, :, None], 0, 255)

    is_chin_pocket = (xx >= 345) & (xx <= 425) & (yy >= 365) & (yy <= 430) & fg & (new_lum < 85.0)
    pocket_w = gaussian_filter(is_chin_pocket.astype(np.float32), sigma=3.0)
    out[:, :, 0] = out[:, :, 0] * (1.0 - 0.70 * pocket_w) + 135.0 * (0.70 * pocket_w)
    out[:, :, 1] = out[:, :, 1] * (1.0 - 0.70 * pocket_w) + 118.0 * (0.70 * pocket_w)
    out[:, :, 2] = out[:, :, 2] * (1.0 - 0.70 * pocket_w) + 98.0 * (0.70 * pocket_w)

    out_rgb = np.where(fg[:, :, None], np.clip(out, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def make_clean_slider(dst_path: Path) -> None:
    """Builds the Pond Slider (120834277) input from RAW_SHARP_DIR / 04_pond_slider.png (no box patch!):
    1. Smoothly trims the tiny bark bump on the upper carapace (x=460..640) along the natural oval dome.
    2. Gently warms the blue-gray sky glare on the wet carapace with a smooth spatial ramp (no seam).
    3. Places a bold crimson-red postorbital ear stripe directly behind the eye across the side of the head
       (spanning ~4.5 SDXL latent pixels vertically) so SDXL renders an unmistakable red ear stripe."""
    src_path = RAW_SHARP_DIR / "04_pond_slider.png"
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    fg = np.any(arr < 250, axis=-1)
    yy, xx = np.ogrid[:1024, :1024]

    dome_y = (
        347.0
        - (347.0 - 316.0) * (xx - 460.0) / (640.0 - 460.0)
        - 6.0 * np.sin(np.clip((xx - 460.0) / 180.0, 0.0, 1.0) * np.pi)
    )
    is_bark_bump = (xx >= 460) & (xx <= 640) & (yy < dome_y)
    arr[is_bark_bump] = 255.0
    fg = fg & (~is_bark_bump)

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    w_shell_x = np.clip((680.0 - xx) / 110.0, 0.0, 1.0) * np.clip((640.0 - yy) / 90.0, 0.0, 1.0)
    sky_glare = w_shell_x * np.clip((b - g + 12.0) / 28.0, 0.0, 1.0) * np.clip((lum - 85.0) / 70.0, 0.0, 1.0) * fg
    arr[:, :, 0] = arr[:, :, 0] * (1.0 - 0.22 * sky_glare)
    arr[:, :, 1] = arr[:, :, 1] * (1.0 - 0.25 * sky_glare)
    arr[:, :, 2] = arr[:, :, 2] * (1.0 - 0.52 * sky_glare)

    # Diagonal postorbital ear patch behind the eye (eye is at x=812, y=182; ear stripe runs from (792, 192) to (742, 238))
    # Distance to segment [(742, 238) -> (792, 192)] on the side of the head:
    vx, vy = 792.0 - 742.0, 192.0 - 238.0
    vlen2 = vx * vx + vy * vy
    t_seg = np.clip(((xx - 742.0) * vx + (yy - 238.0) * vy) / vlen2, 0.0, 1.0)
    px = 742.0 + t_seg * vx
    py = 238.0 + t_seg * vy
    dist_ear = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    w_ear = np.clip((16.0 - dist_ear) / 6.0, 0.0, 1.0) * fg

    arr[:, :, 0] = arr[:, :, 0] * (1.0 - 0.90 * w_ear) + 220.0 * (0.90 * w_ear)
    arr[:, :, 1] = arr[:, :, 1] * (1.0 - 0.90 * w_ear) + 28.0 * (0.90 * w_ear)
    arr[:, :, 2] = arr[:, :, 2] * (1.0 - 0.90 * w_ear) + 28.0 * (0.90 * w_ear)

    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(dst_path)


SPECIES_11_SPEC = [
    (
        "00_northern_shoveler",
        "Northern Shoveler",
        f"animl_gouache_style, a northern shoveler, full body wildlife illustration, single animal with webbed feet visible underneath, dark iridescent green head, bright yellow eye, large spatulate black bill, crisp white breast, warm chestnut-brown belly and flanks, dark slate-gray back, pale blue forewing patch, orange-red legs and webbed feet, {STYLE_V6} --n {NEG_V6} --d 42",
    ),
    (
        "01_northern_pintail",
        "Northern Pintail",
        f"animl_gouache_style, a northern pintail, full body wildlife illustration, single animal with webbed feet visible underneath, rich chocolate-brown head, thin white stripe extending up side of neck, crisp white breast, bluish-gray bill, finely vermiculated gray back and flanks, black-and-buff scapular feathers, long pointed black central tail feathers, gray webbed feet, {STYLE_V6} --n {NEG_V6} --d 42",
    ),
    (
        "02_american_wigeon",
        "American Wigeon",
        f"animl_gouache_style, an american wigeon, full body wildlife illustration, single animal with webbed feet visible underneath, speckled grayish-white head with dark green postorbital patch and pale crown, pale bluish-gray bill with black tip, warm pinkish-brown breast and flanks, brown-and-gray back, crisp white belly and undertail border, gray webbed feet, {STYLE_V6} --n {NEG_V6} --d 42",
    ),
    (
        "03_green-winged_teal",
        "Green-winged Teal",
        f"animl_gouache_style, a green-winged teal, full body wildlife illustration, single animal with webbed feet visible underneath, rich chestnut-brown head with iridescent green eye-patch, vertical white shoulder stripe, dark slate bill, finely vermiculated gray back and flanks, buff-tan breast with fine spots, pale yellow undertail triangle, gray webbed feet, {STYLE_V6} --n {NEG_V6} --d 42",
    ),
    (
        "04_mule_deer",
        "Mule Deer",
        f"animl_gouache_style, a mule deer, full body wildlife illustration, single animal standing on all four slender legs with four dark cloven hooves visible, two front legs and two hind legs separated and distinct, dark eyes, large ears, brown muzzle, smooth matte grayish-brown coat, pale cream rump patch, dark brown bifurcated antlers, black-tipped tail, {STYLE_V6} --n {NEG_V6}, missing leg, three legs, fused legs --d 42",
    ),
    (
        "05_muskrat",
        "Muskrat",
        f"animl_gouache_style, a muskrat, full body wildlife illustration, single animal crouching in profile with rounded body, dark eyes, dark muzzle, uniform warm brown matte coat, darker brown back, lighter tan cheeks, short brown whiskers, {STYLE_V6} --n {NEG_V6} --d 42",
    ),
    (
        "06_california_ground_squirrel",
        "California Ground Squirrel",
        f"animl_gouache_style, a california ground squirrel, full body wildlife illustration, single animal standing in profile on four paws with a single bushy tail, smooth warm buff-brown and gray matte coat, silvery-gray shoulder mantle, finely dappled buff-brown back, smooth buff face with pale eye ring, smooth buff chest and belly, {STYLE_V6} --n {NEG_V6}, two tails, twin tails, split tail, forked tail, acorn, nut, berry, fruit, food, stripes --d 42",
    ),
    (
        "07_american_pika",
        "American Pika",
        f"animl_gouache_style, an american pika, full body wildlife illustration, single animal crouching with rounded hind haunch and four small paws visible, dark eyes, rounded dark ears with pale rims, smooth warm tawny-buff and grayish-brown coat, buff chest, furred paws, {STYLE_V6} --n {NEG_V6} --d 42",
    ),
    (
        "08_pond_slider",
        "Pond Slider",
        f"animl_gouache_style, a red-eared slider, full body wildlife illustration, single animal standing with webbed clawed feet visible, warm olive-brown oval carapace with dark brown scutes, dark olive-green head and legs with thin pale yellow stripes, distinct bright red ear-patch on side of head behind the eye, {STYLE_V6} --n {NEG_V6}, yellow shell, box turtle shell, red legs, red foreleg, red neck, red shell --d 42",
    ),
    (
        "09_common_watersnake",
        "Common Watersnake",
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single snake with a continuous smoothly coiled body from head to tail, dark brown and warm gray-brown scales with dark dorsal crossbands and alternating lateral blotches, pale tan labial scales with dark vertical bars, round pupil, {STYLE_V6} --n {NEG_V6} --d 42",
    ),
    (
        "10_gopher_tortoise",
        "Gopher Tortoise",
        f"animl_gouache_style, a gopher tortoise, full body wildlife illustration, single animal standing on sturdy legs, uniform matte dark brown to grayish-brown oblong domed carapace with concentric scute rings, dark gray-brown scaly head, dark eyes, broad shovel-like scaled forefeet and elephantine hind feet, {STYLE_V6} --n {NEG_V6}, yellow shell, orange markings, box turtle --d 42",
    ),
]


def prepare_all_inputs(in_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    # 00..03: Waterfowl from V5_EVAL / "inputs"
    for stem in ["00_northern_shoveler", "01_northern_pintail", "02_american_wigeon", "03_green-winged_teal"]:
        shutil.copyfile(V5_EVAL / "inputs" / f"{stem}.png", in_dir / f"{stem}.png")
    # 04: Mule Deer from RAW_SHARP_DIR
    shutil.copyfile(RAW_SHARP_DIR / "00_mule_deer.png", in_dir / "04_mule_deer.png")
    # 05: Muskrat (clean chest arch + no extra foot)
    make_clean_muskrat(in_dir / "05_muskrat.png")
    # 06: California Ground Squirrel (zero-blur high-frequency contrast damping)
    make_clean_squirrel(in_dir / "06_california_ground_squirrel.png")
    # 07: American Pika from RAW_SHARP_DIR
    shutil.copyfile(RAW_SHARP_DIR / "03_american_pika.png", in_dir / "07_american_pika.png")
    # 08: Pond Slider (120834277 with clean oval dome + head-aligned crimson postorbital stripe)
    make_clean_slider(in_dir / "08_pond_slider.png")
    # 09: Common Watersnake (15458841 clean dry coiled snake)
    shutil.copyfile(PRE_V6_DIR / "inputs" / "06_watersnake_15458841.png", in_dir / "09_common_watersnake.png")
    # 10: Gopher Tortoise from V5_EVAL / "inputs"
    shutil.copyfile(V5_EVAL / "inputs" / "10_gopher_tortoise.png", in_dir / "10_gopher_tortoise.png")
    print(f"Prepared all 11 clean inputs in {in_dir}", flush=True)


def clean_post(src_path: Path, dst_path: Path) -> Image.Image:
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    im = Image.open(src_path).convert("RGBA")
    cut = remove(im)
    box = subject_bbox(cut)
    if box is None:
        rgb = im.convert("RGB")
        rgb.save(dst_path)
        return rgb
    subj = cut.crop(box)
    scale = int(1024 * 0.78) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    rgb = canvas.convert("RGB")
    rgb.save(dst_path)
    return rgb


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, clean_dir: Path, lora_path: Path, strength: str, cfg: str = "5.5", mul: str = "1.0") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    if len(list(clean_dir.glob("*.png"))) >= len(SPECIES_11_SPEC):
        print(f"  [cached] {clean_dir.name}", flush=True)
        return
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
    for png in sorted(out_dir.glob("*.png")):
        clean_post(png, clean_dir / png.name)


def get_best_v5_image(stem: str) -> Path:
    """Returns our best pre-v6 reference render for each species so we can compare v6 vs v5 side-by-side."""
    mapping = {
        "00_northern_shoveler": V5_EVAL / "clean_step1000_s0.75" / "00_northern_shoveler.png",
        "01_northern_pintail": V5_EVAL / "clean_step1000_s0.75" / "01_northern_pintail.png",
        "02_american_wigeon": V5_EVAL / "clean_step1000_s0.75" / "02_american_wigeon.png",
        "03_green-winged_teal": V5_EVAL / "clean_step1000_s0.75" / "03_green-winged_teal.png",
        "04_mule_deer": PRE_V6_DIR / "col4_s080_step1250" / "00_mule_deer.png",
        "05_muskrat": MS_QUICK_DIR / "col4_s079_step1250" / "00_muskrat.png",
        "06_california_ground_squirrel": PRE_V6_DIR / "col3_s077_step1250" / "02_squirrel.png",
        "07_american_pika": PRE_V6_DIR / "col4_s080_step1250" / "03_pika.png",
        "08_pond_slider": MS_QUICK_DIR / "col2_s074_step1000" / "01_slider.png",
        "09_common_watersnake": PRE_V6_DIR / "col3_s077_step1250" / "06_watersnake_15458841.png",
        "10_gopher_tortoise": V5_EVAL / "clean_step1000_s0.75" / "10_gopher_tortoise.png",
    }
    return mapping[stem]


def build_grid_sheet(rows_subset: list[tuple[str, str, str]], col_defs: list[tuple[str, callable]], out_path: Path) -> None:
    CELL = 360
    HEADER_H = 52
    ROW_LABEL_H = 34
    W = len(col_defs) * CELL
    H = HEADER_H + len(rows_subset) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 17)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    for c_idx, (col_title, _) in enumerate(col_defs):
        draw.text((c_idx * CELL + 12, 15), col_title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, label, _) in enumerate(rows_subset):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), label, fill=(30, 30, 50), font=font_r)
        for c_idx, (_, fn) in enumerate(col_defs):
            p = fn(stem)
            if p.exists():
                # If from PRE_V6_DIR or MS_QUICK_DIR raw output, normalize with clean_post cache
                if "col" in p.parent.name:
                    cached_p = EVAL_DIR / "v5_best_clean" / f"{stem}.png"
                    cached_p.parent.mkdir(parents=True, exist_ok=True)
                    im = clean_post(p, cached_p)
                else:
                    im = Image.open(p).convert("RGB")
                thumb = im.resize((CELL - 8, CELL - 8), Image.LANCZOS)
                sheet.paste(thumb, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    sheet.save(out_path, quality=93)
    print(f"Saved {out_path}", flush=True)


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = EVAL_DIR / "inputs"
    prepare_all_inputs(in_dir)

    pf = EVAL_DIR / "prompts.txt"
    pf.write_text("\n".join(p for _, _, p in SPECIES_11_SPEC), encoding="utf-8")

    if "--prepare-only" in sys.argv:
        return

    lora_1000 = LORA / "output" / "animl_style_v6-step00001000.safetensors"
    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"
    lora_1500 = LORA / "output" / "animl_style_v6.safetensors"

    runs = [
        ("v6_step1000_s075", lora_1000, "0.75"),
        ("v6_step1250_s076", lora_1250, "0.76"),
        ("v6_step1250_s079", lora_1250, "0.79"),
        ("v6_step1500_s078", lora_1500, "0.78"),
    ]
    for tag, ckpt, s in runs:
        if ckpt.exists():
            print(f"\n=== Evaluating {tag} ({ckpt.name}, s={s}) ===", flush=True)
            run_sdxl(in_dir, pf, EVAL_DIR / f"raw_{tag}", EVAL_DIR / f"clean_{tag}", ckpt, s, "5.5", "1.0")

    cols = [
        ("1. Clean Input", lambda stem: in_dir / f"{stem}.png"),
        ("2. Best v5 Baseline", get_best_v5_image),
        ("3. v6 step1000 (s=0.75)", lambda stem: EVAL_DIR / "clean_v6_step1000_s075" / f"{stem}.png"),
        ("4. v6 step1250 (s=0.76)", lambda stem: EVAL_DIR / "clean_v6_step1250_s076" / f"{stem}.png"),
        ("5. v6 step1250 (s=0.79)", lambda stem: EVAL_DIR / "clean_v6_step1250_s079" / f"{stem}.png"),
        ("6. v6 step1500 (s=0.78)", lambda stem: EVAL_DIR / "clean_v6_step1500_s078" / f"{stem}.png"),
    ]

    diag_6 = [row for row in SPECIES_11_SPEC if row[0] in {
        "04_mule_deer", "05_muskrat", "06_california_ground_squirrel",
        "07_american_pika", "08_pond_slider", "09_common_watersnake"
    }]
    waterfowl_4 = SPECIES_11_SPEC[:4]
    land_7 = SPECIES_11_SPEC[4:]

    build_grid_sheet(diag_6, cols, ARTIFACT_DIR / "v6_6species_diagnostic_compare.jpg")
    build_grid_sheet(waterfowl_4, cols, ARTIFACT_DIR / "v6_unseen_waterfowl_compare.jpg")
    build_grid_sheet(land_7, cols, ARTIFACT_DIR / "v6_unseen_land_reptiles_compare.jpg")


if __name__ == "__main__":
    main()
