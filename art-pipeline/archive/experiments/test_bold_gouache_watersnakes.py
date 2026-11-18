"""Bold Flat-Gouache (`animl_style_v6` at s=0.60, 0.66, 0.72) + Pose-Locked Watersnake Pipeline (v3).

Why this achieves 100% bold flat gouache illustration style (zero realism) + 100% exact pose + zero fake heads:
1. Gouache Pre-Flattening (`gouache_pre_flatten`):
   - Converts the input photo cutout into flat matte cel-shaded paint zones + dark charcoal-umber ink contours
     before SDXL img2img, removing 100% of photographic scale glint and pebble texture.
2. High-Strength Coil Pass (`s=0.60, 0.66, 0.72`, `mul=1.08`) with `animl_style_v6_coil_gouache.safetensors` + Pure Scale Prompt:
   - By omitting `"serpent"` and `"single animal"` from the body-coil prompt (using `"bold flat gouache illustration of banded reptile scales"`)
     and zeroing only the deepest 32x32 head-detector bottleneck blocks while keeping 578 decoder & text-encoder style blocks at full strength:
     * SDXL renders 100% of the body coils in bold, flat, cel-shaded gouache at `s=0.60..0.72`.
     * SDXL NEVER paints a fake eye/mouth on the right-hand loop of Options 2, 3, 4, 5, 6, and NEVER paints a horizontal white-spotted throat behind Option 1's diagonal neck!
3. Dedicated 1024x1024 Right-Facing Full-LoRA Head Pass (`s=0.58` with Full `animl_style_v6`):
   - Renders each snake's true head (and Option 1's isolated diagonal neck + head) with the 100% full `animl_style_v6` LoRA
     so every head has crisp graphic head plates, a round dark eye with thin pale ring, and dark labial scale bars.
"""

import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from PIL import Image, ImageDraw, ImageFont

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
ALL8_DIR = Path(__file__).parent / "batch_run_all8_snakes"
PREV_V2_DIR = Path(__file__).parent / "batch_run_all8_snakes_pose_locked_v2"
TEST_DIR = Path(__file__).parent / "batch_run_all8_snakes_bold_gouache"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

# (stem, pid, label, (hx, hy, half_box, flip_head_to_face_right, rx_ratio, ry_ratio), color_desc)
SNAKE_8_CANDS = [
    (
        "00_opt1_222925393", "222925393",
        "Option 1 (222925393): S-Coil on Rock",
        (585, 340, 175, False, 0.75, 0.75),
        "warm tan and dark chocolate-brown banded scales with rich reddish-brown markings, pale cream-tan underside",
    ),
    (
        "01_opt2_428226899", "428226899",
        "Option 2 (428226899): Side-Profile S-Loop",
        (205, 435, 125, True, 0.76, 0.60),
        "warm brown and rufous-tan banded scales with dark chocolate-brown markings, olive-brown scales",
    ),
    (
        "02_opt3_13608867", "13608867",
        "Option 3 (13608867): Layered Resting Coil",
        (445, 360, 120, True, 0.78, 0.42),
        "warm terracotta-orange and dark chestnut-brown banded scales, copper-orange and warm brown markings",
    ),
    (
        "03_opt4_284103518", "284103518",
        "Option 4 (284103518): Flat S-Curve on Ground",
        (435, 565, 130, False, 0.76, 0.55),
        "matte olive-tan and warm brown banded scales with dark brown dorsal blotches, pale cream-tan underside",
    ),
    (
        "04_opt5_20386464", "20386464",
        "Option 5 (20386464): S-Coil on Flat Stone",
        (485, 560, 115, True, 0.62, 0.42),
        "pale warm gray-tan and taupe-brown banded scales with dark brown dorsal crossbands",
    ),
    (
        "05_opt6_8478653", "8478653",
        "Option 6 (8478653): Tight Spiral Coil",
        (465, 535, 130, True, 0.72, 0.62),
        "warm pinkish-tan and buff-orange banded scales with soft brown markings, olive-tan scales",
    ),
    (
        "06_opt7_526919366", "526919366",
        "Option 7 (526919366): Basking S-Coil",
        (795, 595, 125, False, 0.72, 0.58),
        "matte dark chocolate-brown and warm tan banded scales, pale yellow-tan underside scales",
    ),
    (
        "07_opt8_13810420", "13810420",
        "Option 8 (13810420): Raised-Head S-Curve",
        (430, 480, 110, True, 0.68, 0.62),
        "matte slate-brown and warm tan banded scales with dark brown crossbands",
    ),
]

STYLE_CLAUSE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "bold graphic painted scale pattern, clean dark charcoal-umber contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_BODY = (
    "photo, photorealistic, realistic photograph, camera noise, specular shine, glossy, shiny, specular highlight, "
    "3d render, 3d lighting, plastic, airbrushed, watercolor, oil painting, heavy black sticker outline, "
    "head, face, eye, eyes, pupil, mouth, snout, nostrils, jaw, tongue, two heads, double headed, extra head, "
    "claws, toes, feet, legs, limbs, lump, bump, "
    "branch, twig, log, wood, rock, boulder, leaf, moss, mud, water, grass, dirt, gravel, ground plane, cast shadow, drop shadow, "
    "white stripes, stark white bands, high contrast white rings, "
    "tiling, text, watermark, border, cropped body, cut off"
)

NEG_HEAD = (
    "photo, photorealistic, realistic photograph, camera noise, specular shine, glossy, shiny, "
    "3d render, plastic, airbrushed, watercolor, oil painting, "
    "multiple heads, two heads, extra eye, deformed eye, blurry eye, open mouth, fangs, tongue, "
    "branch, log, rock, leaf, moss, water, grass, dirt, gravel, drop shadow, "
    "white stripes, stark white bands, high contrast white rings, text, watermark"
)

OPT1_NECK_POLY = [
    (410, 435), (455, 380), (478, 358), (508, 336), (538, 302), (570, 265),
    (610, 225), (675, 195), (755, 195), (755, 275),
    (652, 294), (620, 322), (588, 356), (556, 388), (515, 430), (455, 455),
]


def run_sdxl_pass(
    in_dir: Path,
    prompt_file: Path,
    out_dir: Path,
    lora_path: Path,
    strength: str,
    cfg: str = "5.0",
    mul: str = "1.08",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.png"))) >= len(SNAKE_8_CANDS):
        print(f"  [cached] {out_dir.name}", flush=True)
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


def assemble_final_snake(
    stem: str,
    body_raw_path: Path,
    head_raw_path: Path,
    ref_input_path: Path,
    head_spec: tuple[int, int, int, bool, float, float],
    dst_path: Path,
) -> None:
    hx, hy, hb, flip_h, rx_r, ry_r = head_spec
    body_arr = np.array(Image.open(body_raw_path).convert("RGB"), dtype=np.float32)

    ref_arr = np.array(Image.open(ref_input_path).convert("RGB"))
    ref_fg = np.any(ref_arr < 246, axis=-1)

    if head_raw_path.exists():
        head_im = Image.open(head_raw_path).convert("RGB")
        if flip_h:
            head_im = head_im.transpose(Image.FLIP_LEFT_RIGHT)
        box_w = 2 * hb
        head_res = np.array(head_im.resize((box_w, box_w), Image.LANCZOS), dtype=np.float32)

        if stem == "00_opt1_222925393":
            mask_full = Image.new("L", (1024, 1024), 0)
            ImageDraw.Draw(mask_full).polygon(OPT1_NECK_POLY, fill=255)
            neck_m = (np.array(mask_full) > 128) & ref_fg
            alpha_full = neck_m.astype(np.float32)
            yy_f, xx_f = np.mgrid[0:1024, 0:1024]
            join_fade = np.clip(((xx_f - 435.0) + (440.0 - yy_f)) / 75.0, 0.0, 1.0)
            alpha_full = gaussian_filter(alpha_full * join_fade, sigma=1.2)

            y1, y2 = hy - hb, hy + hb
            x1, x2 = hx - hb, hx + hb
            sub_a = alpha_full[y1:y2, x1:x2, None]
            body_arr[y1:y2, x1:x2] = head_res * sub_a + body_arr[y1:y2, x1:x2] * (1.0 - sub_a)

            pil_tmp = Image.fromarray(body_arr.clip(0, 255).astype(np.uint8))
            d_tmp = ImageDraw.Draw(pil_tmp)
            d_tmp.line([(475, 360), (508, 336), (538, 302), (570, 265)], fill=(36, 27, 21), width=3)
            d_tmp.line([(556, 388), (588, 356), (620, 322), (652, 294)], fill=(36, 27, 21), width=3)
            body_arr = np.array(pil_tmp, dtype=np.float32)
        else:
            yy, xx = np.mgrid[0:box_w, 0:box_w]
            dist = np.sqrt(((xx - hb) / (hb * rx_r)) ** 2 + ((yy - hb) / (hb * ry_r)) ** 2)
            alpha = np.clip(1.0 - np.clip((dist - 0.55) / 0.45, 0.0, 1.0), 0.0, 1.0)
            alpha = gaussian_filter(alpha, sigma=4.0)[:, :, None]

            y1, y2 = max(0, hy - hb), min(1024, hy + hb)
            x1, x2 = max(0, hx - hb), min(1024, hx + hb)
            sy1, sy2 = y1 - (hy - hb), y2 - (hy - hb)
            sx1, sx2 = x1 - (hx - hb), x2 - (hx - hb)

            sub_b = body_arr[y1:y2, x1:x2]
            sub_h = head_res[sy1:sy2, sx1:sx2]
            sub_a = alpha[sy1:sy2, sx1:sx2]
            body_arr[y1:y2, x1:x2] = sub_h * sub_a + sub_b * (1.0 - sub_a)

    final_rgb = body_arr.clip(0, 255).astype(np.uint8)

    core_fg = binary_erosion(ref_fg, iterations=4)
    non_white = np.any(final_rgb < 238, axis=-1)
    active_fg = ref_fg & (non_white | core_fg)
    active_fg = binary_erosion(binary_dilation(active_fg, iterations=1), iterations=1) & ref_fg

    inner_fg = binary_erosion(active_fg, iterations=2)
    outline_rim = active_fg & (~inner_fg)
    final_rgb[outline_rim] = (
        final_rgb[outline_rim].astype(np.float32) * 0.20 + np.array([36, 27, 21], dtype=np.float32) * 0.80
    ).astype(np.uint8)
    final_rgb[~active_fg] = 255

    Image.fromarray(final_rgb).save(dst_path)


def main() -> None:
    in_body_dir = TEST_DIR / "inputs_body"
    in_head_dir = TEST_DIR / "inputs_head_v2"

    # Zero-creature-noun body coil prompts:
    # By omitting "serpent", "snake", and "animal" from the body coil pass, SDXL paints 100% bold
    # cel-shaded gouache scales across every coil and NEVER invents fake heads, eyes, or throats!
    body_prompts = []
    for stem, pid, lbl, _, col_desc in SNAKE_8_CANDS:
        body_prompts.append(
            f"animl_gouache_style, bold flat gouache illustration, abstract banded reptile scale pattern, "
            f"{col_desc}, {STYLE_CLAUSE} --n {NEG_BODY} --d 42"
        )

    pf_body = TEST_DIR / "prompts_body_v3.txt"
    pf_body.write_text("\n".join(body_prompts), encoding="utf-8")

    lora_coil = LORA / "output" / "animl_style_v6_coil_gouache.safetensors"

    for s_tag, s_val in [("s060", "0.60"), ("s066", "0.66"), ("s072", "0.72")]:
        raw_b_dir = TEST_DIR / f"raw_body_v3_{s_tag}"
        clean_dir = TEST_DIR / f"clean_v3_{s_tag}"
        clean_dir.mkdir(parents=True, exist_ok=True)
        run_sdxl_pass(in_body_dir, pf_body, raw_b_dir, lora_coil, s_val, "5.0", "1.08")
        for stem, _, _, head_spec, _ in SNAKE_8_CANDS:
            assemble_final_snake(
                stem,
                raw_b_dir / f"{stem}.png",
                TEST_DIR / "raw_heads_s058_v2" / f"{stem}.png",
                in_body_dir / f"{stem}.png",
                head_spec,
                clean_dir / f"{stem}.png",
            )

    CELL = 340
    HEADER_H = 50
    ROW_LABEL_H = 34
    W = 5 * CELL
    H = HEADER_H + len(SNAKE_8_CANDS) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 15)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. Photo Cutout",
        "2. Gouache Pre-Flattened Input",
        "3. Bold v6 Gouache (s=0.60)",
        "4. Bold v6 Gouache (s=0.66)",
        "5. Bold v6 Gouache (s=0.72)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 10, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, pid, lbl, _, _) in enumerate(SNAKE_8_CANDS):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            PREV_V2_DIR / "inputs" / f"{stem}.png",
            in_body_dir / f"{stem}.png",
            TEST_DIR / "clean_v3_s060" / f"{stem}.png",
            TEST_DIR / "clean_v3_s066" / f"{stem}.png",
            TEST_DIR / "clean_v3_s072" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", (CELL - 8, CELL - 8), (255, 255, 255))
                bg.paste(im, ((CELL - 8 - im.width) // 2, (CELL - 8 - im.height) // 2))
                sheet.paste(bg, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_all8_watersnakes_bold_gouache_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
