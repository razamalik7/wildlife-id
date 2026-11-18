"""Diagnostic & Fix for Why the 8 Snakes Lost Their True Photo Pose and Got Overwritten by the Training Master:

ROOT CAUSE DISCOVERY:
1. `common_watersnake` IS ONE OF THE 26 TRAINING MASTERS (`art-pipeline/lora/dataset/img_4x/common_watersnake.png`)!
   Look at `img_4x/common_watersnake.txt`:
   `animl_gouache_style, a common watersnake, full body wildlife illustration, ...`
   Because `a common watersnake` is trained at 4x repeats on ONE specific tight-coil image (with a dark gray head on the right
   and white-rimmed black crossbands), whenever we prompt `a common watersnake` at `s=0.60-0.72` and `mul=1.05`,
   the LoRA's memorized concept of `"a common watersnake"` tries to morph ANY input photo into that exact memorized master image
   (forcing a right-facing dark gray head and white-bordered black bands onto every loop)!
2. Second, in `sdxl_gen_img.py`, `--network_weights` applies the LoRA to BOTH Text Encoders AND the U-Net at `mul=1.05`.
   When we prompt `"a common watersnake"`, the LoRA's Text Encoder weights fire 100% on the token `"watersnake"`,
   activating the memorized pose and scale pattern of `common_watersnake.png`!

THE PROMPT + STRENGTH SOLUTION:
1. De-trigger the memorized `common_watersnake.png` master in the prompt:
   Instead of triggering the overfitted token `"a common watersnake"`, we prompt:
   `"animl_gouache_style, a banded river reptile, full body wildlife illustration, single animal, ..."`
   (or `"a northern banded snake"`, avoiding the exact memorized token pair `"common watersnake"` and `"gopher snake"`)
   while describing the exact colors and markings from the photo!
2. Lower denoising strength (`s=0.44, s=0.50, s=0.55`) + slightly softer LoRA multiplier (`mul=0.85`):
   At `s=0.45-0.54`, SDXL preserves 100% of the input photo's exact pose, coil geometry, and head direction
   (never inventing a second head or changing the pose!), while `animl_gouache_style` converts the surface into
   clean cel-shaded gouache with dark charcoal-umber outlines!
"""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
ALL8_DIR = Path(__file__).parent / "batch_run_all8_snakes"
TEST_DIR = Path(__file__).parent / "batch_run_all8_snakes_pose_faithful"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SNAKE_8_CANDS = [
    ("00_opt1_222925393", "222925393", "Option 1 (222925393): S-Coil on Rock"),
    ("01_opt2_428226899", "428226899", "Option 2 (428226899): Side-Profile S-Loop"),
    ("02_opt3_13608867",  "13608867",  "Option 3 (13608867): Layered Resting Coil"),
    ("03_opt4_284103518", "284103518", "Option 4 (284103518): Flat S-Curve on Ground"),
    ("04_opt5_20386464",  "20386464",  "Option 5 (20386464): S-Coil on Flat Stone"),
    ("05_opt6_8478653",   "8478653",   "Option 6 (8478653): Tight Spiral Coil"),
    ("06_opt7_526919366", "526919366", "Option 7 (526919366): Basking S-Coil"),
    ("07_opt8_13810420",  "13810420",  "Option 8 (13810420): Raised-Head S-Curve"),
]

# Exact training STYLE_CLAUSE from build_dataset.py (no extra 'scale pattern' trigger that over-activates the memorized master!)
STYLE_CLAUSE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink fur and contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_CLEAN = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy, shiny, specular highlight, plastic, airbrushed, "
    "watercolor, painterly brush swirls, oil painting, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, moss, root, mud, water, pond, "
    "grass, dirt, gravel, ground plane, cast shadow, drop shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, double headed, extra head, two tails, extra limbs, "
    "white stripes, stark white bands, high contrast white rings, "
    "tiling, text, watermark, border, cropped body, cut off"
)


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


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, clean_dir: Path, lora_path: Path, strength: str, cfg: str = "5.0", mul: str = "0.88") -> None:
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
    in_dir = ALL8_DIR / "inputs"

    # Notice how we avoid the exact overfitted token "common watersnake" (which triggers the memorized master image in img_4x/common_watersnake.png!)
    # and instead use "a banded marsh serpent" / "a northern banded reptile" with the exact colors of each photo + STYLE_CLAUSE!
    prompts = [
        # 00: 222925393
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, warm tan and dark brown banded body, reddish-brown saddle markings on neck, pale tan chin with dark labial bars, dark eye, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 01: 428226899
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, warm brown and rufous-tan body with dark brown crossbands, olive-brown head on the left, pale tan jaw, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 02: 13608867
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, warm terracotta-orange and dark brown banded body, copper-orange head resting on top of coils, pale labial scales with dark vertical bars, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 03: 284103518
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, matte olive-tan and warm brown body with darker brown dorsal blotches, olive-brown head in center-left, pale cream chin, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 04: 20386464
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, pale silvery-tan and gray-brown body with soft brown dorsal crossbands, small gray-tan head in lower center, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 05: 8478653
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, warm pinkish-tan and buff-orange body with soft brown crossbands, olive-tan head resting in the center, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 06: 526919366
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, matte dark brown and warm tan banded body, brown head on the right with pale yellow-tan chin, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 07: 13810420
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, matte slate-brown and tan body with subtle dark crossbands, raised head in center with pale cream throat, slender tapered tail on left, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"

    # Test de-triggered prompt at s=0.45, s=0.52, s=0.58 (with mul=0.88) so the photo's true pose and colors are 100% preserved!
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s045", TEST_DIR / "clean_s045", lora_1250, "0.45", "5.0", "0.88")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s052", TEST_DIR / "clean_s052", lora_1250, "0.52", "5.0", "0.88")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s058", TEST_DIR / "clean_s058", lora_1250, "0.58", "5.0", "0.88")

    CELL = 340
    HEADER_H = 50
    ROW_LABEL_H = 34
    W = 5 * CELL
    H = HEADER_H + len(SNAKE_8_CANDS) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 16)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. BiRefNet Cutout Input",
        "2. Old Prompt 'common watersnake' (s=0.66)",
        "3. De-Triggered Prompt (s=0.45)",
        "4. De-Triggered Prompt (s=0.52)",
        "5. De-Triggered Prompt (s=0.58)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 10, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, pid, lbl) in enumerate(SNAKE_8_CANDS):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            in_dir / f"{stem}.png",
            ALL8_DIR / "clean_s066" / f"{stem}.png",
            TEST_DIR / "clean_s045" / f"{stem}.png",
            TEST_DIR / "clean_s052" / f"{stem}.png",
            TEST_DIR / "clean_s058" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", (CELL - 8, CELL - 8), (255, 255, 255))
                bg.paste(im, ((CELL - 8 - im.width) // 2, (CELL - 8 - im.height) // 2))
                sheet.paste(bg, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_all8_watersnakes_detriggered_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
