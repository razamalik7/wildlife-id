"""Fixes the 'fused figure-8' impossible geometry on the uncropped Common Watersnake (15458841):
Exact coordinates verified from snake_crossing_zoom_grid.jpg:
1. Why the raw photo turned into a flat merged '8' in SDXL:
   - In the raw photo, at the crossing `(x=645..740, y=380..540)` where the neck curves OVER the lower-right coil,
     and at `(x=465..605, y=460..585)` where the head/jaw rests OVER the lower-left coil, the brown scales of the
     upper layer and lower layer have almost identical pixel values (no dark boundary line between them).
   - When SDXL's 8x VAE downsamples the image, those un-outlined crossings fuse into a single flat '8' junction!
   - Furthermore, the snake's tail is hidden underneath the crossing in the photo, so the outer silhouette has no visible tail tip.
2. Three clean, anatomically exact solutions tested side-by-side:
   - Variant A (Uncropped 3D-Layered Figure-8): Keeps 100% of the uncropped coil and places crisp dark charcoal-umber
     contour lines + cast occlusion shadows along the exact anatomical overlap edges:
     * Right neck edge `(740, 385) -> (725, 450) -> (695, 505) -> (635, 545)` where the neck crosses ON TOP OF the right loop.
     * Left neck edge `(650, 415) -> (615, 455) -> (575, 470)` where the neck rises OVER the middle coil.
     * Head/cheek/chin contour `(565, 470) -> (485, 480) -> (468, 525) -> (510, 575) -> (605, 588) -> (680, 565)` where the head rests ON TOP OF both lower coils.
   - Variant B (3D-Layered Figure-8 + Visible Pointed Tail Tip): Same exact 3D over/under neck & head layering at the crossing,
     plus the end of the lower-right loop `(x=565..635, y=580..680)` tapers into a clean pointed tail tip under the chin
     (with a clean white gap separating the tail tip from the left coil) so you can trace 1 continuous body from Head -> Neck (OVER) -> Left Loop -> Under Head -> Right Loop -> Pointed Tail Tip!
"""

import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
PRE_V6_DIR = Path(__file__).parent / "batch_run_pre_v6_diag"
TEST_DIR = Path(__file__).parent / "batch_run_snake_layering"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

STYLE_SNAKE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "crisp scale pattern and dark charcoal-umber contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_SNAKE = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy, shiny, specular highlight, plastic, airbrushed, "
    "watercolor, painterly brush swirls, oil painting, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, drop shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, double headed, two tails, extra limbs, merged body, fused coils, flat figure eight, "
    "tiling, text, watermark, border, cropped body, cut off"
)


def make_snake_crossing_variants(in_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    s_raw = Image.open(PRE_V6_DIR / "inputs" / "06_watersnake_15458841.png").convert("RGB")
    arr_base = np.array(s_raw, dtype=np.float32)
    fg = np.any(arr_base < 248, axis=-1)
    yy, xx = np.ogrid[:1024, :1024]

    lum = 0.299 * arr_base[:, :, 0] + 0.587 * arr_base[:, :, 1] + 0.114 * arr_base[:, :, 2]
    lum_loc = gaussian_filter(lum, sigma=12.0)
    new_lum = np.clip(lum_loc + (lum - lum_loc) * 1.60, 32.0, 195.0)
    scale = new_lum / np.maximum(lum, 1.0)
    arr = np.where(fg[:, :, None], np.clip(arr_base * scale[:, :, None], 0, 255), 255.0)

    # Exact polygon of the UPPER LAYER (Head + Neck curving from top arch down into the head)
    # Verified directly from snake_crossing_zoom_grid.jpg:
    upper_poly = [
        (648, 350), (738, 365), (740, 415), (726, 462), (702, 502),
        (665, 535), (692, 552), (682, 575), (645, 588), (585, 590),
        (525, 578), (482, 550), (466, 515), (476, 480), (518, 466),
        (575, 470), (618, 458), (646, 425), (648, 385)
    ]
    upper_mask_im = Image.new("L", (1024, 1024), 0)
    ImageDraw.Draw(upper_mask_im).polygon(upper_poly, fill=255)
    upper_fg = (np.array(upper_mask_im.filter(ImageFilter.GaussianBlur(1.5))) > 128) & fg

    # Also define the MIDDLE LAYER at the crossing: the coil segment `(x=565..655, y=420..495)`
    # that passes UNDER the right neck `(x=645..735)` and UNDER the top of the head `(y=470)`!
    dist_outside_upper = distance_transform_edt(~upper_fg)

    arr_a = arr.copy()
    # 1. Gently brighten the upper head + neck (+12%) so it sits visually in front of the underlying coils
    in_crossing_box = (xx >= 450) & (xx <= 765) & (yy >= 380) & (yy <= 625)
    arr_a = np.where((upper_fg & in_crossing_box)[:, :, None], np.clip(arr_a * 1.12 + 8.0, 0, 232), arr_a)

    # 2. Cast occlusion shadow onto the LOWER coils right where they pass UNDER the head and neck (width 28px)
    under_shadow = fg & (~upper_fg) & (dist_outside_upper <= 30.0) & in_crossing_box
    sh_w = np.clip((30.0 - dist_outside_upper) / 30.0, 0.0, 1.0)
    arr_a = np.where(
        under_shadow[:, :, None],
        arr_a * (1.0 - 0.65 * (sh_w[:, :, None] ** 1.2)) + np.array([26.0, 22.0, 20.0])[None, None, :] * (0.65 * (sh_w[:, :, None] ** 1.2)),
        arr_a,
    )

    # 3. Draw crisp dark charcoal-umber overlap contour lines ONLY along the internal overlap seams
    #    (where the upper neck/head lies on top of the lower coils):
    seam_im = Image.new("L", (1024, 1024), 0)
    d_seam = ImageDraw.Draw(seam_im)
    # Seam 1: Right edge of neck crossing over the right-hand loop:
    d_seam.line([(738, 395), (728, 448), (705, 496), (665, 535)], fill=255, width=7)
    # Seam 2: Left edge of neck + top/left/bottom edge of head resting on top of the lower coils:
    d_seam.line([
        (648, 412), (620, 455), (578, 470), (520, 466), (476, 480),
        (466, 515), (482, 550), (525, 578), (585, 590), (645, 586)
    ], fill=255, width=7)
    seam_mask = (np.array(seam_im.filter(ImageFilter.GaussianBlur(0.8))) > 90) & fg

    # Also add a subtle pale rim just inside the upper neck/head edge so SDXL's VAE sees a crisp stepped edge
    rim_im = Image.new("L", (1024, 1024), 0)
    d_rim = ImageDraw.Draw(rim_im)
    d_rim.line([(731, 395), (721, 446), (698, 492), (660, 529)], fill=255, width=4)
    d_rim.line([
        (654, 414), (626, 458), (580, 476), (522, 472), (482, 485), (473, 515)
    ], fill=255, width=4)
    rim_mask = (np.array(rim_im.filter(ImageFilter.GaussianBlur(0.8))) > 90) & fg & (~seam_mask)

    arr_a[ rim_mask ] = np.clip(arr_a[ rim_mask ] * 1.22 + 18.0, 0, 235)
    arr_a[ seam_mask ] = [26.0, 22.0, 20.0]

    Image.fromarray(arr_a.astype(np.uint8)).save(in_dir / "00_snake_layered_over_under.png")

    # Variant B: Same 3D Over/Under Neck & Head Layering + Smoothly Tapered Tail Tip on the Lower-Right Coil!
    # Look at the exact y-bounds of the lower-right coil at x=730: y_top=636, y_bot=719.
    # For the left-hand coil passing under the head (x=510..565), its bottom edge is at y=605 (at x=550).
    # So we only carve the lower-right coil for (x >= 568, y >= 590), leaving the left coil's smooth curve under the jaw 100% untouched!
    arr_b = arr_a.copy()
    in_tail_zone = (xx <= 730) & (yy >= 590) & (xx >= 568)
    t_tail = np.clip((xx - 585.0) / (730.0 - 585.0), 0.0, 1.0)
    y_top_curve = 662.0 + (636.0 - 662.0) * (t_tail ** 0.90)
    y_bot_curve = 662.0 + (719.0 - 662.0) * (t_tail ** 0.90)
    keep_tail = (xx >= 585) & (yy >= y_top_curve) & (yy <= y_bot_curve)

    fg_b = np.where(in_tail_zone, fg & keep_tail, fg)
    missing_tip = in_tail_zone & keep_tail & (~fg)
    for y_px, x_px in zip(*np.where(missing_tip)):
        src_x = min(750, 715 + (715 - x_px) // 3)
        src_y = min(710, max(645, y_px))
        arr_b[y_px, x_px] = arr_a[src_y, src_x]
    fg_b = fg_b | missing_tip
    arr_b[~fg_b] = 255.0

    dist_b = distance_transform_edt(fg_b)
    tail_rim = fg_b & (dist_b <= 3.0) & in_tail_zone & (xx <= 720)
    arr_b[tail_rim] = [34.0, 28.0, 25.0]

    Image.fromarray(arr_b.astype(np.uint8)).save(in_dir / "01_snake_true_3d_coil_with_tail.png")


def clean_post(src_path: Path, dst_path: Path) -> Image.Image:
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    im = Image.open(src_path).convert("RGBA")
    cut = remove(im)
    arr = np.array(cut)
    a = arr[:, :, 3]
    arr[a < 180, 3] = 0
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
    if len(list(clean_dir.glob("*.png"))) >= 2:
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
    make_snake_crossing_variants(in_dir)

    prompts = [
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single coiled snake with its head and neck clearly overlapping on top of the lower body coil, distinct 3d over-under coil layering with cast shadow under the neck crossing, warm reddish-brown and gray-brown scales with dark brown dorsal crossbands, pale labial scales with dark vertical bars, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single coiled snake with its head and neck overlapping on top of the lower body coil and a tapered pointed tail tip at the bottom right, distinct over-under coil layering, warm reddish-brown and gray-brown scales with dark brown dorsal crossbands, pale labial scales with dark vertical bars, round pupil, {STYLE_SNAKE} --n {NEG_SNAKE} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"

    run_sdxl(in_dir, pf, TEST_DIR / "raw_s058", TEST_DIR / "clean_s058", lora_1250, "0.58", "5.5", "1.05")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s064", TEST_DIR / "clean_s064", lora_1250, "0.64", "5.5", "1.05")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s070", TEST_DIR / "clean_s070", lora_1250, "0.70", "5.5", "1.05")

    rows = [
        ("00_snake_layered_over_under", "Variant A: Uncropped Coil + Crisp Over/Under Neck & Head Contour Layering"),
        ("01_snake_true_3d_coil_with_tail", "Variant B: Over/Under Neck Layering + Un-fused Pointed Tail Tip Under Chin"),
    ]

    CELL = 380
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
        "1. 3D-Layered Input",
        "2. v6 (s=0.58)",
        "3. v6 (s=0.64)",
        "4. v6 (s=0.70)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, lbl) in enumerate(rows):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            in_dir / f"{stem}.png",
            TEST_DIR / "clean_s058" / f"{stem}.png",
            TEST_DIR / "clean_s064" / f"{stem}.png",
            TEST_DIR / "clean_s070" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB").resize((CELL - 8, CELL - 8), Image.LANCZOS)
                sheet.paste(im, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_snake_3d_layering_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
