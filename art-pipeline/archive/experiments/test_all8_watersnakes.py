"""Runs all 8 clean Common Watersnake reference candidates through BiRefNet cutout normalization
and animl_style_v6 (at s=0.60, s=0.66, s=0.72), then builds a side-by-side comparison sheet:
Col 1: Original Photo | Col 2: BiRefNet Cutout Input | Col 3: v6 (s=0.60) | Col 4: v6 (s=0.66) | Col 5: v6 (s=0.72).
"""

import urllib.request
import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
FC = Path(__file__).parent / "vetting" / "funnel_cache"
TEST_DIR = Path(__file__).parent / "batch_run_all8_snakes"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SNAKE_8_CANDS = [
    ("00_opt1_222925393", "222925393", "Option 1 (222925393): S-Coil on Rock — Head + Pointed Tail Visible"),
    ("01_opt2_428226899", "428226899", "Option 2 (428226899): Side-Profile S-Loop on Moss"),
    ("02_opt3_13608867",  "13608867",  "Option 3 (13608867): Layered Resting Coil on Log — Head on Top"),
    ("03_opt4_284103518", "284103518", "Option 4 (284103518): Flat S-Curve on Ground — Zero Crossings"),
    ("04_opt5_20386464",  "20386464",  "Option 5 (20386464): S-Coil on Flat Stone — Full Body & Tail"),
    ("05_opt6_8478653",   "8478653",   "Option 6 (8478653): Tight Spiral Coil on Concrete — Head on Top"),
    ("06_opt7_526919366", "526919366", "Option 7 (526919366): Basking S-Coil on Log — Uncrossed Loops"),
    ("07_opt8_13810420",  "13810420",  "Option 8 (13810420): Raised-Head S-Curve on Gravel — Full Body to Tail"),
]

STYLE_SNAKE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "crisp scale pattern and dark charcoal-umber contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_SNAKE = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy, shiny, specular highlight, plastic, airbrushed, "
    "watercolor, painterly brush swirls, oil painting, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, moss, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, gravel, ground plane, cast shadow, drop shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, double headed, two tails, extra limbs, merged body, fused coils, "
    "tiling, text, watermark, border, cropped body, cut off"
)


def fetch_large_or_cached(pid: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / f"{pid}.jpg"
    if dst.exists():
        return dst
    # Try fetching large (1024px) from iNaturalist S3 first, fallback to cached medium
    for ext in ["jpg", "jpeg", "png"]:
        url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/large.{ext}"
        try:
            urllib.request.urlretrieve(url, dst)
            if dst.stat().st_size > 10000:
                return dst
        except Exception:
            pass
    cached = FC / f"{pid}.jpg"
    if cached.exists():
        dst.write_bytes(cached.read_bytes())
    return dst


def cutout_and_normalize(src_path: Path, dst_path: Path) -> Image.Image:
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    im = Image.open(src_path).convert("RGBA")
    cut = remove(im)
    arr = np.array(cut)
    # Clean soft background alpha halo
    arr[arr[:, :, 3] < 140, 3] = 0
    cut = Image.fromarray(arr)
    box = subject_bbox(cut)
    if box is None:
        rgb = im.convert("RGB").resize((1024, 1024), Image.LANCZOS)
        rgb.save(dst_path)
        return rgb
    subj = cut.crop(box)
    scale = int(1024 * 0.78) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    rgb = canvas.convert("RGB")

    # Gentle anti-shine + local band clarity (no blurring, no cropping!)
    c_arr = np.array(rgb, dtype=np.float32)
    fg = np.any(c_arr < 248, axis=-1)
    lum = 0.299 * c_arr[:, :, 0] + 0.587 * c_arr[:, :, 1] + 0.114 * c_arr[:, :, 2]
    lum_loc = gaussian_filter(lum, sigma=10.0)
    new_lum = np.clip(lum_loc + (lum - lum_loc) * 1.25, 28.0, 215.0)
    scale_l = new_lum / np.maximum(lum, 1.0)
    out = np.where(fg[:, :, None], np.clip(c_arr * scale_l[:, :, None], 0, 255).astype(np.uint8), 255)
    out_im = Image.fromarray(out)
    out_im.save(dst_path)
    return out_im


def clean_post(src_path: Path, dst_path: Path) -> Image.Image:
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    im = Image.open(src_path).convert("RGBA")
    cut = remove(im)
    arr = np.array(cut)
    arr[arr[:, :, 3] < 165, 3] = 0
    cut = Image.fromarray(arr)
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


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, clean_dir: Path, lora_path: Path, strength: str, cfg: str = "5.5", mul: str = "1.05") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    if len(list(clean_dir.glob("*.png"))) >= len(SNAKE_8_CANDS):
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


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = TEST_DIR / "raw_photos"
    in_dir = TEST_DIR / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    for stem, pid, _ in SNAKE_8_CANDS:
        photo_p = fetch_large_or_cached(pid, raw_dir)
        cutout_and_normalize(photo_p, in_dir / f"{stem}.png")
        print(f"Prepared cutout: {stem}.png", flush=True)

    prompts = [
        # 00: 222925393 (S-coil on rock)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single S-coiled snake with one raised head at the top and a pointed tail tip at the bottom right, dark brown and warm tan scales with bold dark brown dorsal crossbands and alternating lateral blotches, pale labial scales with dark vertical bars, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        # 01: 428226899 (S-loop on moss)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single snake in a side-profile S-loop with one head on the left and a tapered tail at the bottom, warm brown and reddish-brown scales with dark brown dorsal crossbands, pale labial scales with dark bars, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        # 02: 13608867 (Layered resting coil)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single coiled snake resting its head on top of its stacked body coils, warm reddish-orange and dark brown banded scales, pale labial scales with dark vertical bars, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        # 03: 284103518 (Flat S-curve on ground)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single snake in an uncrossed S-curve with one head in the center-left and a continuous body, olive-brown and tan scales with dark brown dorsal blotches and lateral bands, pale labial scales with dark vertical bars, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        # 04: 20386464 (S-coil on flat stone)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single coiled snake with one head in the lower center and a continuous body tapering to a thin tail at the bottom left, pale gray-tan scales with dark brown dorsal crossbands and blotches, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        # 05: 8478653 (Tight spiral coil on concrete)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single coiled snake with its head and neck resting across the center of the coil, warm reddish-tan and buff scales with darker crossbands, pale labial scales, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        # 06: 526919366 (Basking S-coil on log)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single snake in a side-profile S-coil with one head on the right, dark brown and warm tan scales with dark dorsal crossbands, pale labial scales with dark bars, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        # 07: 13810420 (Raised-head S-curve on gravel)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single snake in a loose S-curve with raised head in the center and a long tapered tail tip extending to the left, dark gray-brown and tan scales with dark dorsal bands, pale chin and labial scales, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"

    run_sdxl(in_dir, pf, TEST_DIR / "raw_s060", TEST_DIR / "clean_s060", lora_1250, "0.60", "5.5", "1.05")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s066", TEST_DIR / "clean_s066", lora_1250, "0.66", "5.5", "1.05")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s072", TEST_DIR / "clean_s072", lora_1250, "0.72", "5.5", "1.05")

    CELL = 340
    HEADER_H = 50
    ROW_LABEL_H = 34
    W = 5 * CELL
    H = HEADER_H + len(SNAKE_8_CANDS) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 17)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. Original Photo",
        "2. BiRefNet Cutout Input",
        "3. v6 (s=0.60)",
        "4. v6 (s=0.66)",
        "5. v6 (s=0.72)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, pid, lbl) in enumerate(SNAKE_8_CANDS):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            raw_dir / f"{pid}.jpg",
            in_dir / f"{stem}.png",
            TEST_DIR / "clean_s060" / f"{stem}.png",
            TEST_DIR / "clean_s066" / f"{stem}.png",
            TEST_DIR / "clean_s072" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB")
                # Fit preserving aspect ratio inside (CELL-8, CELL-8)
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", (CELL - 8, CELL - 8), (255, 255, 255))
                bg.paste(im, ((CELL - 8 - im.width) // 2, (CELL - 8 - im.height) // 2))
                sheet.paste(bg, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_all8_watersnakes_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
