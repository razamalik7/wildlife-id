"""Clean unblurred v5 baseline across all 6 test species using SHARP BiRefNet cutouts (zero blurring/posterization)
with our cleaned prompts:
- CFG = 5.5
- Matte/ink style prompt (no 'gouache')
- Pond Slider (120834277): prompt describes 'dark olive-brown oval carapace with dark brown scutes' (NO 'yellow-cream rings'!)
- Muskrat: dangling plant root under front paw trimmed on the raw cutout, and unseen 'scaled tail, webbed hind feet' removed from prompt
- Mammal markings sanitized ('fur' -> 'matte coat', 'mottled' -> 'two-tone flat', 'dusky hairs' -> 'sparse dark ink marks')

Columns:
- Col 1: Sharp Raw Cutout Input
- Col 2: v5-step1000 (s=0.74, CFG=5.5) on Sharp Raw Cutout
- Col 3: v5-step1250 (s=0.77, CFG=5.5) on Sharp Raw Cutout
- Col 4: v5-step1250 (s=0.80, CFG=5.5) on Sharp Raw Cutout
"""

import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
PREV_1AB_DIR = Path(__file__).parent / "batch_run_1ab_2_test"
TEST_DIR = Path(__file__).parent / "batch_run_raw_unblurred_6sp"
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
    "white fur streaks, silver hair highlights, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, two tails, extra limbs, extra feet, fused legs, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_INK_MATTE = (
    "clean stylized 2d wildlife illustration, flat matte cel-shaded color zones with hard-edged stepped shading, "
    "solid uniform flat color blocks with sparse crisp dark ink linework and dark ink contour marks on top, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, zero soft gradients, "
    "isolated on a plain solid white background"
)


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
    in_dir = TEST_DIR / "inputs_sharp"
    in_dir.mkdir(exist_ok=True)

    raw_sources = {
        "mule_deer": PREV_1AB_DIR / "inputs_raw" / "00_mule_deer.png",
        "muskrat": PREV_1AB_DIR / "inputs_raw" / "01_muskrat.png",
        "california_ground_squirrel": PREV_1AB_DIR / "inputs_raw" / "02_california_ground_squirrel.png",
        "american_pika": PREV_1AB_DIR / "inputs_raw" / "03_american_pika.png",
        "pond_slider": PREV_1AB_DIR / "inputs_slider" / "01_slider_120834277.png",
        "common_watersnake": V5_EVAL / "inputs" / "09_common_watersnake.png",
    }

    for i, (slug, _) in enumerate(SPECIES):
        dst = in_dir / f"{i:02d}_{slug}.png"
        if not dst.exists():
            im = Image.open(raw_sources[slug]).convert("RGB")
            if slug == "muskrat":
                # Trim only the dangling plant root under the front paw (zero blurring!)
                arr = np.array(im)
                yy, xx = np.ogrid[:1024, :1024]
                arr[(xx < 310) & (yy > 688)] = [255, 255, 255]
                Image.fromarray(arr).save(dst)
            else:
                im.save(dst)

    prompts = [
        # 0. Mule Deer
        f"animl_gouache_style, a mule deer, full body wildlife illustration, single animal standing on all four slender legs with dark cloven hooves visible, dark eyes, large ears, brown muzzle, grayish-brown matte coat, pale white rump patch, dark brown bifurcated antlers, slender brown legs, black-tipped tail, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 1. Muskrat
        f"animl_gouache_style, a muskrat, full body wildlife illustration, single animal with small front paws tucked neatly under chin, dark eyes, dark muzzle, uniform warm brown matte coat, darker brown back, lighter tan cheeks and belly, short brown whiskers, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 2. California Ground Squirrel
        f"animl_gouache_style, a california ground squirrel, full body wildlife illustration, single animal with rounded hind haunch and paws visible, pale cream eye ring, two-tone flat gray-brown dorsal coat zone, lighter gray shoulders, buff underside, stylized tail with dark ink marks, dark paws, sparse dark ink line marks, light brown sides, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 3. American Pika
        f"animl_gouache_style, an american pika, full body wildlife illustration, single animal with rounded hind haunch and paws visible, dark eyes, whiskered muzzle, grayish-brown rounded body, warm buff chest, flat matte coat, rounded dark ears with pale rims, clean paws, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
        # 4. Pond Slider (NO 'yellow-cream rings' on shell!)
        f"animl_gouache_style, a pond slider, full body wildlife illustration, single animal standing with webbed clawed feet and short tail clearly visible, dark olive-brown oval carapace with dark brown scutes, dark olive-green head and limbs with bright yellow longitudinal stripes and a distinct red stripe behind the eye, {STYLE_INK_MATTE} --n {NEG_INK_MATTE}, yellow shell, box turtle shell --d 42",
        # 5. Common Watersnake
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single animal coiled cleanly, dark brown body with dark crossbands, reddish-brown head scales, dark eyes, pale cream-gray belly scales with dark crescents, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42",
    ]
    pfile = TEST_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    lora_1000 = LORA / "output" / "animl_style_v5-step00001000.safetensors"
    lora_1250 = LORA / "output" / "animl_style_v5-step00001250.safetensors"

    print("Running Col 2 (Sharp Raw Input + Clean Prompts, step1000, s=0.74)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col2_s074_step1000", lora_1000, "0.74", "5.5", "1.0")

    print("Running Col 3 (Sharp Raw Input + Clean Prompts, step1250, s=0.77)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col3_s077_step1250", lora_1250, "0.77", "5.5", "1.0")

    print("Running Col 4 (Sharp Raw Input + Clean Prompts, step1250, s=0.80)...")
    run_sdxl(in_dir, pfile, TEST_DIR / "col4_s080_step1250", lora_1250, "0.80", "5.5", "1.0")

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
        "1. Sharp Raw Cutout Input",
        "2. Raw + Clean Prompt (step1000, s=0.74)",
        "3. Raw + Clean Prompt (step1250, s=0.77)",
        "4. Raw + Clean Prompt (step1250, s=0.80)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 16), title, fill=(20, 20, 30), font=font_b)

    in_imgs = sorted(in_dir.glob("*.png"))
    c2_imgs = sorted((TEST_DIR / "col2_s074_step1000").glob("*.png"))
    c3_imgs = sorted((TEST_DIR / "col3_s077_step1250").glob("*.png"))
    c4_imgs = sorted((TEST_DIR / "col4_s080_step1250").glob("*.png"))

    for r_idx, (slug, common) in enumerate(SPECIES):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), f"{common} ({slug})", fill=(30, 30, 50), font=font_r)

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

    out_sheet = ARTIFACT_DIR / "raw_unblurred_6sp_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}")


if __name__ == "__main__":
    main()
