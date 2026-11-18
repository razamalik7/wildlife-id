"""Pure SDXL + animl_style_v6 run with anatomy / anti-double-head / anti-bulge tokens
placed in Chunk 1 (<75 tokens) so they enter SDXL's pooled negative embedding (`uncond_pool`)
and receive explicit token weighting `(1.4-1.5)`.
Zero OpenCV filters or post-processing hacks.
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
PREV_DIR = Path(__file__).parent / "batch_run_all8_snakes_pose_faithful"
TEST_DIR = Path(__file__).parent / "batch_run_all8_snakes_anatomy_neg"
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

# Keep positive prompt concise so all anatomy + style tokens stay in Chunk 1 (<75 tokens)
ANATOMY_POS = (
    "(single head only:1.35), (one head and one tapered tail:1.3), "
    "smooth continuous cylindrical snake coils, even body thickness with gradual smooth taper toward tail,"
)

STYLE_CLAUSE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

# CRITICAL FIX: Put anti-double-head & anti-bulge/warping tokens FIRST (within the 75-token Chunk 1 limit
# so SDXL's pooled `uncond_pool` embedding actually sees them!) and weight them strongly.
NEG_ANATOMY_FIRST = (
    "(two heads:1.5), (double headed:1.5), (multiple heads:1.5), (extra head:1.5), (head on tail end:1.5), "
    "(weird bulges:1.4), (lumpy swollen body:1.4), (unrealistic shape:1.4), (non-smooth coils:1.4), "
    "(impossible geometry:1.4), (disconnected or fused coils:1.4), (uneven pinched tapering:1.4), "
    "(warped deformed anatomy:1.4), (claws, legs, limbs, toes:1.4), (open mouth, tongue:1.3), "
    "photo, photorealistic, glossy, specular shine, stark white stripes, white rings"
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


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, clean_dir: Path, lora_path: Path, strength: str, cfg: str = "5.5", mul: str = "0.88") -> None:
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
        "--max_embeddings_multiples", "2",
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

    prompts = [
        # 00: 222925393
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} warm tan and dark brown banded body, reddish-brown saddle markings, pale tan chin with dark labial bars, dark round eye, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
        # 01: 428226899
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} warm brown and rufous-tan body with dark brown crossbands, single olive-brown head on left, tapered tail loop below, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
        # 02: 13608867
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} warm terracotta-orange and dark brown banded body, single copper-orange head resting on upper right of coils, dark round eye, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
        # 03: 284103518
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} matte olive-tan and warm brown body with darker brown dorsal blotches, single olive-brown head in center-left, smooth body bend on right, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
        # 04: 20386464
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} pale silvery-tan and gray-brown body with soft brown crossbands, single small gray-tan head in lower center, smooth coiled body, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
        # 05: 8478653
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} warm pinkish-tan and buff-orange body with soft brown crossbands, single olive-tan head resting in center, smooth outer coil loop, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
        # 06: 526919366
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} matte dark brown and warm tan banded body, single brown head on bottom right with pale yellow-tan chin, smooth coiled body on left, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
        # 07: 13810420
        f"animl_gouache_style, a banded marsh serpent, {ANATOMY_POS} matte slate-brown and tan body with subtle dark crossbands, single raised head in center, smooth S-curve body tapering to slender tail tip on left, {STYLE_CLAUSE} --n {NEG_ANATOMY_FIRST} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"

    # Run s=0.45 and s=0.50 with CFG=5.5 so the negative prompt is enforced more strongly
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s045", TEST_DIR / "clean_s045", lora_1250, "0.45", "5.5", "0.88")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s050", TEST_DIR / "clean_s050", lora_1250, "0.50", "5.5", "0.88")

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
        "1. BiRefNet Cutout Input",
        "2. Prev Col 3 (s=0.45, old neg)",
        "3. Prev Col 4 (s=0.52, old neg)",
        "4. NEW Anatomy Neg (s=0.45)",
        "5. NEW Anatomy Neg (s=0.50)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 10, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, pid, lbl) in enumerate(SNAKE_8_CANDS):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            in_dir / f"{stem}.png",
            PREV_DIR / "clean_s045" / f"{stem}.png",
            PREV_DIR / "clean_s052" / f"{stem}.png",
            TEST_DIR / "clean_s045" / f"{stem}.png",
            TEST_DIR / "clean_s050" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", (CELL - 8, CELL - 8), (255, 255, 255))
                bg.paste(im, ((CELL - 8 - im.width) // 2, (CELL - 8 - im.height) // 2))
                sheet.paste(bg, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_all8_watersnakes_anatomy_neg_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
