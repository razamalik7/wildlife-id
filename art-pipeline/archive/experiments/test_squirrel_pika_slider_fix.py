"""Clean test fixing the 3 bugs from test_squirrel_pika_slider_fix.py:
1. Remove 'like american marten' from STYLE_INK_MATTE (which caused SDXL to turn the Pond Slider into a furry marten and the Pika into two animals!).
2. California Ground Squirrel:
   - Use the exact `inputs_v2/02_california_ground_squirrel.png` body (which had ONLY 1 tail and clean body texture in flat_v2_6species_compare.jpg Col 3/4),
   - Fix ONLY the face/head texture by filling the white-hair holes on the forehead (`binary_fill_holes`) and tightening `anat_drop` from `lum_3 < 85` to `lum_3 < 45` (so only the eye pupil/nostril/claws are dark, leaving the shaded cheek/face smooth instead of blotchy!).
3. Pond Slider (120834277):
   - Zero rectangular box masks! Use global highlight clamping (`lum <= 165` on low-saturation glare) on `01_slider_120834277.png`, and run with `step1000` (`s=0.73` and `s=0.75`) WITHOUT 'yellow-cream rings' in the prompt!
"""

import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import percentile_filter, gaussian_filter, binary_fill_holes
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
PREV_1AB_DIR = Path(__file__).parent / "batch_run_1ab_2_test"
PREV_V2_DIR = Path(__file__).parent / "batch_run_flat_v2_6sp"
TEST_DIR = Path(__file__).parent / "batch_run_sps_clean"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

NEG_INK_MATTE = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy shell, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, soft blended brushwork, soft color blending, oil painting, blurry fur, mottled fur, "
    "white fur streaks, silver hair highlights, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, two tails, extra limbs, fused legs, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_INK_MATTE = (
    "clean stylized 2d wildlife illustration, flat matte cel-shaded color zones with hard-edged stepped shading, "
    "solid uniform flat color blocks with sparse crisp dark ink linework and dark ink contour marks on top, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, zero soft gradients, "
    "isolated on a plain solid white background"
)


def make_squirrel_face_fixed(src_path: Path, dst_path: Path) -> None:
    """Identical to make_flat_v2_mammal from test_flat_v2_6species.py (which gave the great body & single tail in Col 3/4),
    EXCEPT:
    1. `fg = binary_fill_holes(np.any(arr < 252, axis=-1))` so bright white forehead hairs (>=250) aren't skipped!
    2. `anat_drop` requires `lum_3 < 46.0` (instead of `< 85.0`) so it only re-injects the eye, nostril, and claws—NOT the shaded cheek/face fur!"""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = binary_fill_holes(np.any(arr < 252, axis=-1))

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Replace blown-out white forehead/neck hairs (lum > 185) with local 30th-percentile warm gray-brown before scaling
    med_color = np.stack([percentile_filter(arr[:, :, c], percentile=30, size=45) for c in range(3)], axis=-1)
    arr_clean = np.where((lum > 185.0)[:, :, None] & fg[:, :, None], med_color, arr)
    lum_clean = np.maximum(0.299 * arr_clean[:, :, 0] + 0.587 * arr_clean[:, :, 1] + 0.114 * arr_clean[:, :, 2], 1.0)

    lum_undercoat = percentile_filter(lum_clean, percentile=25, size=65)
    clamped_lum = np.minimum(np.minimum(lum_clean, lum_undercoat + 20.0), 162.0)
    scale = clamped_lum / lum_clean
    suppressed = np.clip(arr_clean * scale[:, :, None], 0, 255)

    med_rgb = np.stack([percentile_filter(suppressed[:, :, c], percentile=50, size=17) for c in range(3)], axis=-1)
    smooth_rgb = np.stack([gaussian_filter(med_rgb[:, :, c], sigma=5.5) for c in range(3)], axis=-1)

    s_lum = np.maximum(0.299 * smooth_rgb[:, :, 0] + 0.587 * smooth_rgb[:, :, 1] + 0.114 * smooth_rgb[:, :, 2], 1.0)
    q_lum = np.round(s_lum / 24.0) * 24.0
    flat_zones = np.clip(smooth_rgb * (q_lum / s_lum)[:, :, None], 0, 255)

    # ONLY re-inject deep pinpoint darks (`lum_3 < 46.0` = eye, nostril, mouth cleft, claws), leaving the shaded face smooth!
    lum_3 = gaussian_filter(lum, sigma=1.4)
    lum_nb = percentile_filter(lum_3, percentile=60, size=21)
    anat_drop = np.where(lum_3 < 46.0, np.clip(lum_nb - lum_3 - 14.0, 0, 255), 0.0) / 255.0
    dark_mult = np.clip(1.0 - anat_drop * 1.5, 0.08, 1.0)[:, :, None]

    out_rgb = np.where(fg[:, :, None], np.clip(flat_zones * dark_mult, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def make_clean_slider_no_box(src_path: Path, dst_path: Path) -> None:
    """De-glare low-saturation sky reflection across the turtle shell smoothly (no rectangular box!)
    while keeping high-saturation red ear stripe & yellow neck stripes 100% untouched."""
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / np.maximum(max_c, 1.0)

    # Low-saturation bright pixels (sky reflection on carapace, or weathered pale scute patches) get shifted to warm dark olive-brown
    glare_w = np.clip((lum - 85.0) / 55.0, 0.0, 1.0) * np.clip((0.35 - sat) / 0.20, 0.0, 1.0)
    warm_brown = np.stack([
        np.full_like(r, 92.0),
        np.full_like(g, 80.0),
        np.full_like(b, 54.0),
    ], axis=-1)
    out = arr * (1.0 - glare_w[:, :, None]) + warm_brown * glare_w[:, :, None]

    # Enhance crimson red postorbital stripe (where R > G + 12) so it never gets lost
    is_red = (r > g + 12.0) & (r > b + 15.0) & fg
    out[is_red, 0] = np.clip(out[is_red, 0] * 1.30, 150, 225)
    out[is_red, 1] = out[is_red, 1] * 0.70
    out[is_red, 2] = out[is_red, 2] * 0.70

    out_rgb = np.where(fg[:, :, None], np.clip(out, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, lora_path: Path, strength: str, cfg: str, mul: str) -> None:
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= 1:
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
    in_sq = TEST_DIR / "in_sq"
    in_sl = TEST_DIR / "in_sl"
    in_sq.mkdir(exist_ok=True)
    in_sl.mkdir(exist_ok=True)

    make_squirrel_face_fixed(
        PREV_1AB_DIR / "inputs_raw" / "02_california_ground_squirrel.png",
        in_sq / "00_squirrel.png",
    )
    make_clean_slider_no_box(
        PREV_1AB_DIR / "inputs_slider" / "01_slider_120834277.png",
        in_sl / "00_slider.png",
    )

    # Exact same Squirrel prompt structure that gave 1 single tail in flat_v2_6species_compare.jpg!
    p_sq = (
        "animl_gouache_style, a california ground squirrel, full body wildlife illustration, single animal "
        "with rounded hind haunch and paws visible, pale cream eye ring, two-tone flat gray-brown dorsal coat zone, "
        "lighter gray shoulders, buff underside, stylized tail with dark ink marks, dark paws, sparse dark ink line marks, light brown sides, "
        f"{STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
    )
    pf_sq = TEST_DIR / "prompt_sq.txt"
    pf_sq.write_text(p_sq, encoding="utf-8")

    # Pond Slider prompt WITHOUT 'yellow-cream rings' on the shell!
    neg_sl = NEG_INK_MATTE + ", yellow shell, box turtle shell, ornate yellow shell rings"
    p_sl = (
        "animl_gouache_style, a pond slider, full body wildlife illustration, single animal "
        "standing with webbed clawed feet and short tail clearly visible, "
        "dark olive-brown oval carapace with dark brown scutes, "
        "dark olive-green head and limbs with bright yellow longitudinal stripes and a distinct red stripe behind the eye, "
        f"{STYLE_INK_MATTE} --n {neg_sl} --d 42"
    )
    pf_sl = TEST_DIR / "prompt_sl.txt"
    pf_sl.write_text(p_sl, encoding="utf-8")

    lora_1000 = LORA / "output" / "animl_style_v5-step00001000.safetensors"
    lora_1250 = LORA / "output" / "animl_style_v5-step00001250.safetensors"

    # Run Squirrel at s=0.78 (step1250) and s=0.80 (step1250)
    run_sdxl(in_sq, pf_sq, TEST_DIR / "sq_s078", lora_1250, "0.78", "5.5", "1.0")
    run_sdxl(in_sq, pf_sq, TEST_DIR / "sq_s080", lora_1250, "0.80", "5.5", "1.0")

    # Run Pond Slider at step1000 (s=0.72 and s=0.75, mul=1.0)
    run_sdxl(in_sl, pf_sl, TEST_DIR / "sl_s072_step1000", lora_1000, "0.72", "5.5", "1.0")
    run_sdxl(in_sl, pf_sl, TEST_DIR / "sl_s075_step1000", lora_1000, "0.75", "5.5", "1.0")

    CELL = 440
    HEADER_H = 55
    ROW_LABEL_H = 36
    W = 4 * CELL
    H = HEADER_H + 2 * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 18)
        font_r = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. Fixed Input",
        "2. Prev Best (from flat_v2 sheet)",
        "3. Fixed Run A (sq s=0.78 / slider s=0.72)",
        "4. Fixed Run B (sq s=0.80 / slider s=0.75)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 16), title, fill=(20, 20, 30), font=font_b)

    rows = [
        (
            "California Ground Squirrel (Face fur smoothed via lum_3<46 + forehead hole filled)",
            Image.open(in_sq / "00_squirrel.png").convert("RGB"),
            clean_post(sorted((PREV_V2_DIR / "col3_s078_step1250").glob("*squirrel*.png"))[0]),
            clean_post(sorted((TEST_DIR / "sq_s078").glob("*.png"))[0]),
            clean_post(sorted((TEST_DIR / "sq_s080").glob("*.png"))[0]),
        ),
        (
            "Pond Slider (De-glared brown shell + NO 'yellow rings' in prompt + step1000)",
            Image.open(in_sl / "00_slider.png").convert("RGB"),
            clean_post(sorted((PREV_1AB_DIR / "slider_outs").glob("*120834277*.png"))[0]),
            clean_post(sorted((TEST_DIR / "sl_s072_step1000").glob("*.png"))[0]),
            clean_post(sorted((TEST_DIR / "sl_s075_step1000").glob("*.png"))[0]),
        ),
    ]

    for r_idx, (label, c1, c2, c3, c4) in enumerate(rows):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), label, fill=(30, 30, 50), font=font_r)
        for c_idx, im in enumerate([c1, c2, c3, c4]):
            thumb = im.resize((CELL - 8, CELL - 8), Image.LANCZOS)
            sheet.paste(thumb, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle(
                [c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL],
                outline=(210, 215, 220), width=1
            )

    out_sheet = ARTIFACT_DIR / "sq_slider_clean_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}")


if __name__ == "__main__":
    main()
