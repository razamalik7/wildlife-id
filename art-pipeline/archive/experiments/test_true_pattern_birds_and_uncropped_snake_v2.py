"""Fixes the American Wigeon head color & white wing patch + tests slight edge-preservedfeather contour enhancement
without any synthetic scallop grid, plus tests Common Watersnake with uncropped coil + subtle shadow gap between loops.
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
PRE_V6_DIR = Path(__file__).parent / "batch_run_pre_v6_diag"
EVAL_DIR = Path(__file__).parent / "batch_run_v6_eval"
TEST_DIR = Path(__file__).parent / "batch_run_true_birds_snake_v2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

STYLE_BIRD_NATURAL = (
    "muted matte natural colors, hand-painted gouache wildlife illustration with crisp painted feather edges "
    "and subtle warm brown-umber ink feather contour lines, stepped matte shading, "
    "zero gloss or shine, clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

STYLE_SNAKE_NATURAL = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "crisp scale pattern and dark charcoal-umber contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_CLEAN = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy, shiny, specular highlight, plastic, airbrushed, "
    "watercolor, painterly brush swirls, oil painting, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, drop shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, double headed, two tails, extra limbs, "
    "tiling, text, watermark, border, cropped body, cut off, waterline cutoff"
)


def make_paddling_and_tucked_feet() -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    teal_gen = Image.open(EVAL_DIR / "clean_v6_step1250_s076" / "03_green-winged_teal.png").convert("RGB")
    feet_crop = teal_gen.crop((450, 657, 560, 725))
    f_arr = np.array(feet_crop, dtype=np.float32)
    f_fg = np.any(f_arr < 242, axis=-1)
    f_lum = 0.299 * f_arr[:, :, 0] + 0.587 * f_arr[:, :, 1] + 0.114 * f_arr[:, :, 2]

    f_gray_rgb = np.stack([
        np.where(f_fg, np.clip(f_lum * 0.72 + 15, 25, 165), 255),
        np.where(f_fg, np.clip(f_lum * 0.75 + 18, 25, 170), 255),
        np.where(f_fg, np.clip(f_lum * 0.78 + 22, 28, 175), 255),
    ], axis=-1).astype(np.uint8)
    f_rgba = Image.fromarray(np.dstack([f_gray_rgb, (f_fg * 255).astype(np.uint8)]), mode="RGBA")

    # 1. Paddling feet: rotated -38 degrees (toes pointing backward/down-left as if paddling in water)
    paddling = f_rgba.rotate(-38, resample=Image.BICUBIC, expand=True)
    pad_bg = Image.new("RGBA", paddling.size, (255, 255, 255, 0))
    pad_comp = Image.alpha_composite(pad_bg, paddling)

    # 2. Tucked swimming feet: rotated -55 degrees and warm gray-brown, tucked under rear belly
    tucked = f_rgba.resize((96, 58), Image.LANCZOS).rotate(-55, resample=Image.BICUBIC, expand=True)
    tuck_bg = Image.new("RGBA", tucked.size, (255, 255, 255, 0))
    tuck_comp = Image.alpha_composite(tuck_bg, tucked)

    return pad_comp.convert("RGB"), pad_comp.split()[3], tuck_comp.convert("RGB"), tuck_comp.split()[3]


def natural_feather_sharpen(arr: np.ndarray, fg: np.ndarray, boost: float = 1.35) -> np.ndarray:
    """Amplifies ONLY the bird's real high-frequency feather edges (sigma=3.5) without any synthetic grid,
    while clamping extreme specular glare (> 232) and deep black crush (< 26)."""
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lum_loc = gaussian_filter(lum, sigma=3.5)
    detail = np.clip((lum - lum_loc) * boost, -28.0, 22.0)
    base_lum = np.clip(lum_loc, 28.0, 228.0)
    new_lum = np.clip(base_lum + detail, 24.0, 235.0)
    scale = new_lum / np.maximum(lum, 1.0)
    return np.where(fg[:, :, None], np.clip(arr * scale[:, :, None], 0, 255).astype(np.uint8), 255)


def prepare_inputs(in_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    pad_rgb, pad_mask, tuck_rgb, tuck_mask = make_paddling_and_tucked_feet()
    yy, xx = np.ogrid[:1024, :1024]

    # 00: Northern Shoveler (real feather edge boost, zero synthetic grid)
    shov_im = Image.open(V5_EVAL / "inputs" / "00_northern_shoveler.png").convert("RGB")
    shov_arr = np.array(shov_im, dtype=np.float32)
    shov_fg = np.any(shov_arr < 248, axis=-1)
    Image.fromarray(natural_feather_sharpen(shov_arr, shov_fg, 1.35)).save(in_dir / "00_northern_shoveler.png")

    # 01: Northern Pintail (real feather edge boost + rounded lower belly + PADDLING webbed feet)
    p_im = Image.open(V5_EVAL / "inputs" / "01_northern_pintail.png").convert("RGB")
    p_arr = np.array(p_im, dtype=np.float32)
    p_fg = np.any(p_arr < 248, axis=-1)
    belly_bot_p = 635.0 + 44.0 * np.sin(np.clip((xx - 285.0) / (715.0 - 285.0), 0.0, 1.0) * np.pi)
    belly_top_p = 602.0 + 18.0 * np.sin(np.clip((xx - 285.0) / (715.0 - 285.0), 0.0, 1.0) * np.pi)
    in_belly_p = (xx >= 285) & (xx <= 715) & (yy >= belly_top_p) & (yy <= belly_bot_p)
    w_p = np.clip((yy - belly_top_p) / np.maximum(belly_bot_p - belly_top_p, 1.0), 0.0, 1.0)
    belly_col_p = np.array([218.0, 214.0, 206.0])[None, None, :] * (1.0 - 0.12 * (w_p[:, :, None] ** 2))
    p_arr = np.where((in_belly_p & (~p_fg))[:, :, None], belly_col_p, p_arr)
    p_arr = np.where((in_belly_p & p_fg)[:, :, None], p_arr * (1.0 - (w_p[:, :, None] ** 1.8)) + belly_col_p * (w_p[:, :, None] ** 1.8), p_arr)
    rim_p = (yy >= belly_bot_p - 3.0) & (yy <= belly_bot_p) & (xx >= 292) & (xx <= 710)
    p_arr[rim_p] = [65.0, 60.0, 55.0]
    p_fg = p_fg | in_belly_p
    p_clean = Image.fromarray(natural_feather_sharpen(p_arr, p_fg, 1.35))
    p_clean.paste(pad_rgb, (420, 652), pad_mask)
    p_clean.save(in_dir / "01_northern_pintail.png")

    # 02: American Wigeon (real feather edge boost + rounded lower belly + TUCKED swimming feet)
    w_im = Image.open(V5_EVAL / "inputs" / "02_american_wigeon.png").convert("RGB")
    w_arr = np.array(w_im, dtype=np.float32)
    w_fg = np.any(w_arr < 248, axis=-1)
    belly_bot_w = 574.0 + 48.0 * np.sin(np.clip((xx - 305.0) / (762.0 - 305.0), 0.0, 1.0) * np.pi)
    belly_top_w = 556.0 + 12.0 * np.sin(np.clip((xx - 305.0) / (762.0 - 305.0), 0.0, 1.0) * np.pi)
    in_belly_w = (xx >= 305) & (xx <= 762) & (yy >= belly_top_w) & (yy <= belly_bot_w)
    w_w = np.clip((yy - belly_top_w) / np.maximum(belly_bot_w - belly_top_w, 1.0), 0.0, 1.0)
    belly_col_w = np.array([232.0, 228.0, 222.0])[None, None, :] * (1.0 - 0.10 * (w_w[:, :, None] ** 2))
    w_arr = np.where((in_belly_w & (~w_fg))[:, :, None], belly_col_w, w_arr)
    w_arr = np.where((in_belly_w & p_fg if False else (in_belly_w & w_fg))[:, :, None], w_arr * (1.0 - (w_w[:, :, None] ** 1.6)) + belly_col_w * (w_w[:, :, None] ** 1.6), w_arr)
    rim_w = (yy >= belly_bot_w - 3.0) & (yy <= belly_bot_w) & (xx >= 312) & (xx <= 756)
    w_arr[rim_w] = [65.0, 60.0, 55.0]
    w_fg = w_fg | in_belly_w
    w_clean = Image.fromarray(natural_feather_sharpen(w_arr, w_fg, 1.35))
    w_clean.paste(tuck_rgb, (395, 592), tuck_mask)
    w_clean.save(in_dir / "02_american_wigeon.png")

    # 03: Green-winged Teal (real feather edge boost + standing feet)
    t_im = Image.open(V5_EVAL / "inputs" / "03_green-winged_teal.png").convert("RGB")
    t_arr = np.array(t_im, dtype=np.float32)
    t_fg = np.any(t_arr < 248, axis=-1)
    t_clean = Image.fromarray(natural_feather_sharpen(t_arr, t_fg, 1.35))
    teal_gen = Image.open(EVAL_DIR / "clean_v6_step1250_s076" / "03_green-winged_teal.png").convert("RGB")
    feet_crop = teal_gen.crop((450, 657, 560, 725))
    f_arr = np.array(feet_crop, dtype=np.float32)
    f_fg = np.any(f_arr < 242, axis=-1)
    t_clean.paste(feet_crop, (460, 625), Image.fromarray((f_fg * 255).astype(np.uint8)))
    t_clean.save(in_dir / "03_green-winged_teal.png")

    # 04: Common Watersnake (15458841 UNCROPPED full coil with clear dark boundary lines between overlapping loops!)
    s_raw = Image.open(PRE_V6_DIR / "inputs" / "06_watersnake_15458841.png").convert("RGB")
    arr_s = np.array(s_raw, dtype=np.float32)
    fg_s = np.any(arr_s < 248, axis=-1)
    lum_s = 0.299 * arr_s[:, :, 0] + 0.587 * arr_s[:, :, 1] + 0.114 * arr_s[:, :, 2]
    lum_s_loc = gaussian_filter(lum_s, sigma=12.0)
    new_lum_s = np.clip(lum_s_loc + (lum_s - lum_s_loc) * 1.65, 28.0, 195.0)
    scale_s = new_lum_s / np.maximum(lum_s, 1.0)
    Image.fromarray(np.where(fg_s[:, :, None], np.clip(arr_s * scale_s[:, :, None], 0, 255).astype(np.uint8), 255)).save(in_dir / "04_watersnake_15458841_uncropped.png")


def clean_post(src_path: Path, dst_path: Path) -> Image.Image:
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    im = Image.open(src_path).convert("RGBA")
    cut = remove(im)
    arr = np.array(cut)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    yy, xx = np.ogrid[:1024, :1024]
    is_ground_shadow = (yy > 640) & (xx < 435) & (r > 195) & (g > 195) & (b > 195)
    arr[is_ground_shadow | (a < 180), 3] = 0
    cut = Image.fromarray(arr)
    box = subject_bbox(cut)
    if box is None:
        rgb = im.convert("RGB")
        rgb.save(dst_path)
        return rgb
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
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
    if len(list(clean_dir.glob("*.png"))) >= 5:
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
    in_dir = TEST_DIR / "inputs"
    prepare_inputs(in_dir)

    prompts = [
        # 00: Northern Shoveler
        f"animl_gouache_style, a northern shoveler, full body wildlife illustration, single flying duck with orange feet, dark iridescent green head, bright yellow eye, long black spatulate bill, white breast and neck collar, rufous-chestnut belly, pale blue and white wing covert patches, {STYLE_BIRD_NATURAL} --n {NEG_CLEAN}, spotted belly --d 42",
        # 01: Northern Pintail (paddling feet + true photo markings)
        f"animl_gouache_style, a northern pintail, full body wildlife illustration, single duck with paddling gray webbed feet underneath, solid chocolate-brown head, white breast extending as a thin white stripe up the back of the neck, bluish-gray bill, finely vermiculated gray flanks, long black-and-buff scapular feathers on back, pointed tail, {STYLE_BIRD_NATURAL} --n {NEG_CLEAN}, wood duck, chest spot --d 42",
        # 02: American Wigeon (tucked swimming feet + TRUE photo markings: explicitly NO solid green head in negative!)
        f"animl_gouache_style, an american wigeon, full body wildlife illustration, single duck with tucked swimming webbed feet underneath, (bright cream-white forehead crown and emerald-green eye stripe from eye to nape:1.25), speckled buff-tan cheeks and neck, pale blue-gray bill with black tip, warm pinkish-brown breast and flanks with a bold white wing patch, black undertail coverts, {STYLE_BIRD_NATURAL} --n {NEG_CLEAN}, wood duck --d 42",
        # 03: Green-winged Teal (true photo markings)
        f"animl_gouache_style, a green-winged teal, full body wildlife illustration, single standing duck with webbed feet underneath, solid rufous-chestnut head with iridescent green eye-patch behind eye, dark slate bill, vertical white stripe in front of wing, gray flanks and back, buff breast with fine dark speckles, pale yellow undertail triangle, {STYLE_BIRD_NATURAL} --n {NEG_CLEAN}, wood duck, white face stripe, white cheek --d 42",
        # 04: Common Watersnake (UNCROPPED 15458841 full coil!)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single coiled snake resting its head near the center, warm reddish-brown and gray-brown scales with dark brown dorsal crossbands, pale labial scales with dark vertical bars, round pupil, {STYLE_SNAKE_NATURAL} --n {NEG_CLEAN} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"

    run_sdxl(in_dir, pf, TEST_DIR / "raw_s060", TEST_DIR / "clean_s060", lora_1250, "0.60", "5.5", "1.05")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s066", TEST_DIR / "clean_s066", lora_1250, "0.66", "5.5", "1.05")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s071", TEST_DIR / "clean_s071", lora_1250, "0.71", "5.5", "1.05")

    rows = [
        ("00_northern_shoveler", "Northern Shoveler (True Photo Patterning + Natural Feather Edges)"),
        ("01_northern_pintail", "Northern Pintail (Paddling Feet + True Photo Patterning)"),
        ("02_american_wigeon", "American Wigeon (Tucked Feet + Cream Crown & Green Eye-Patch)"),
        ("03_green-winged_teal", "Green-winged Teal (Standing Feet + True Chestnut/Green Head & Vertical Bar)"),
        ("04_watersnake_15458841_uncropped", "Common Watersnake (Uncropped 15458841 Full Coil — Single Head, Zero Cut Artifacts)"),
    ]

    CELL = 360
    HEADER_H = 50
    ROW_LABEL_H = 34
    W = 4 * CELL
    H = HEADER_H + len(rows) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 17)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. True Photo Input (Paddling/Tucked Feet)",
        "2. v6 (s=0.60)",
        "3. v6 (s=0.66)",
        "4. v6 (s=0.71)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, lbl) in enumerate(rows):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            in_dir / f"{stem}.png",
            TEST_DIR / "clean_s060" / f"{stem}.png",
            TEST_DIR / "clean_s066" / f"{stem}.png",
            TEST_DIR / "clean_s071" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB").resize((CELL - 8, CELL - 8), Image.LANCZOS)
                sheet.paste(im, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_true_pattern_birds_and_uncropped_snake_v2.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
