"""Test why the art style at s=0.45 ('banded marsh serpent') looked too photographic/3D,
and unlock the TRUE flat cel-shaded gouache snake style from `gopher_snake_1.png` / `gopher_snake_2.png`:
1. In v5/v6, `common_watersnake.png` (an AI-generated 3D-scale glossy snake) was added to the dataset,
   while `gopher_snake_1.png` and `gopher_snake_2.png` are the TRUE flat cel-shaded snake masters.
2. In `animl_style_v4-step00001250.safetensors`, `common_watersnake.png` was NOT in the dataset—only
   the true flat cel-shaded masters (`gopher_snake_1.png` and `gopher_snake_2.png`)!
3. Also, in both v4 and v6, triggering the trained flat-style snake token (`a gopher snake` with
   watersnake crossband colors) activates the exact flat cel-shaded brushstroke + dark ink outline
   style of `gopher_snake_1.png` and `gopher_snake_2.png` instead of realistic 3D photo scales!
"""

import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
ALL8_DIR = Path(__file__).parent / "batch_run_all8_snakes"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")
TEST_DIR = Path(__file__).parent / "batch_run_true_art_style_3_7_8"


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


def run_pass(in_dir: Path, pf: Path, out_dir: Path, clean_dir: Path, lora_path: Path, strength: str, cfg: str = "6.5", mul: str = "1.0") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    if len(list(clean_dir.glob("*.png"))) >= 3:
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
        "--from_file", str(pf),
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
    in_dir.mkdir(parents=True, exist_ok=True)

    # Prepare shadow-free inputs for Option 3, Option 7, and Option 8
    shutil.copyfile(ALL8_DIR / "inputs" / "02_opt3_13608867.png", in_dir / "00_opt3_13608867.png")

    # Option 7 input with the bottom-right log shadow removed (same exact curve as 06_opt7_526919366_s045_no_shadow_a.png)
    im7 = Image.open(ALL8_DIR / "inputs" / "06_opt7_526919366.png").convert("RGB")
    arr7 = np.array(im7)
    for x in range(620, 910):
        if x < 765:
            t = (765 - x) / 145.0
            y_lim = int(round(662 - 34 * (t ** 1.6)))
        else:
            t = (x - 765) / 128.0
            y_lim = int(round(662 - 68 * (t ** 2.0)))
        y_lim = max(560, min(800, y_lim))
        arr7[y_lim:, x, :] = 255
    Image.fromarray(arr7).save(in_dir / "01_opt7_526919366.png")

    # Option 8 input with the rightmost gravel shadow bump removed
    im8 = Image.open(ALL8_DIR / "inputs" / "07_opt8_13810420.png").convert("RGB")
    arr8 = np.array(im8)
    for y in range(440, 580):
        t = (y - 495) / 55.0
        x_lim = int(round(862 - 14.0 * (t ** 2)))
        arr8[y, x_lim:, :] = 255
    Image.fromarray(arr8).save(in_dir / "02_opt8_13810420.png")

    NEG = (
        "(two heads:1.5), (double headed:1.5), (multiple heads:1.5), (extra head:1.5), "
        "(weird bulges:1.4), (lumpy swollen body:1.4), (unrealistic shape:1.4), (non-smooth coils:1.4), "
        "(open mouth, tongue:1.4), photo, photorealistic, 3d render, glossy, shiny, specular highlight, "
        "3d scales, stark white stripes, white rings, cast shadow, drop shadow"
    )

    # 1. V4 LoRA prompts (uses V4 training style clause)
    STYLE_V4 = (
        "rich natural colors, crisp distinct painted color steps, flat stylized "
        "gouache texture marks, clean thin dark charcoal-umber outline, "
        "isolated on a plain solid white background"
    )
    prompts_v4 = [
        f"a gopher snake, full body wildlife illustration, (single head only:1.35), warm terracotta-brown and dark brown banded snake body, single copper-brown head resting on coils, pale cream labial scales with dark vertical bars, {STYLE_V4} --n {NEG} --d 42",
        f"a gopher snake, full body wildlife illustration, (single head only:1.35), matte warm tan and dark brown banded snake body, single brown head on bottom right with pale tan chin and dark labial bars, {STYLE_V4} --n {NEG} --d 42",
        f"a gopher snake, full body wildlife illustration, (single head only:1.35), matte warm slate-tan and brown banded snake body, single raised head in center with pale cream throat, slender tapered tail on left, {STYLE_V4} --n {NEG} --d 42",
    ]
    pf_v4 = TEST_DIR / "prompts_v4.txt"
    pf_v4.write_text("\n".join(prompts_v4), encoding="utf-8")

    # 2. V6 LoRA prompts (activating the `a gopher snake` flat-cel-shaded master style in v6 with watersnake banding)
    STYLE_V6 = (
        "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
        "subtle low-contrast warm brown-umber ink contour lines, "
        "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
    )
    prompts_v6 = [
        f"animl_gouache_style, a gopher snake, full body wildlife illustration, (single head only:1.35), warm terracotta-brown and dark brown crossbanded body, single copper-brown head resting on coils, pale cream labial scales with dark bars, {STYLE_V6} --n {NEG} --d 42",
        f"animl_gouache_style, a gopher snake, full body wildlife illustration, (single head only:1.35), warm sandy-tan and dark brown crossbanded body, single brown head on bottom right with pale tan chin, {STYLE_V6} --n {NEG} --d 42",
        f"animl_gouache_style, a gopher snake, full body wildlife illustration, (single head only:1.35), warm tan and dark brown crossbanded body, single raised head in center with cream throat, slender tapered tail on left, {STYLE_V6} --n {NEG} --d 42",
    ]
    pf_v6 = TEST_DIR / "prompts_v6.txt"
    pf_v6.write_text("\n".join(prompts_v6), encoding="utf-8")

    lora_v4 = LORA / "output" / "animl_style_v4-step00001250.safetensors"
    lora_v6 = LORA / "output" / "animl_style_v6-step00001250.safetensors"

    run_pass(in_dir, pf_v4, TEST_DIR / "raw_v4_s054", TEST_DIR / "clean_v4_s054", lora_v4, "0.54", "6.5", "1.0")
    run_pass(in_dir, pf_v4, TEST_DIR / "raw_v4_s062", TEST_DIR / "clean_v4_s062", lora_v4, "0.62", "6.5", "1.0")
    run_pass(in_dir, pf_v6, TEST_DIR / "raw_v6_s052", TEST_DIR / "clean_v6_s052", lora_v6, "0.52", "6.0", "1.0")
    run_pass(in_dir, pf_v6, TEST_DIR / "raw_v6_s058", TEST_DIR / "clean_v6_s058", lora_v6, "0.58", "6.0", "1.0")

    # Build comparison sheet
    CELL = 360
    HEADER_H = 50
    ROW_LABEL_H = 34
    rows = [
        ("00_opt3_13608867", "Option 3 (13608867): Layered Resting Coil"),
        ("01_opt7_526919366", "Option 7 (526919366): Basking S-Coil (Shadow Removed)"),
        ("02_opt8_13810420", "Option 8 (13810420): Raised-Head S-Curve (Shadow Removed)"),
    ]
    cols = [
        ("1. Shadow-Free Input", in_dir),
        ("2. v4 Flat Gouache (s=0.54)", TEST_DIR / "clean_v4_s054"),
        ("3. v4 Flat Gouache (s=0.62)", TEST_DIR / "clean_v4_s062"),
        ("4. v6 Gopher-Style (s=0.52)", TEST_DIR / "clean_v6_s052"),
        ("5. v6 Gopher-Style (s=0.58)", TEST_DIR / "clean_v6_s058"),
    ]
    W = len(cols) * CELL
    H = HEADER_H + len(rows) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 15)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    for c_idx, (title, _) in enumerate(cols):
        draw.text((c_idx * CELL + 10, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, lbl) in enumerate(rows):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        for c_idx, (_, dpath) in enumerate(cols):
            p = dpath / f"{stem}.png"
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", (CELL - 8, CELL - 8), (255, 255, 255))
                bg.paste(im, ((CELL - 8 - im.width) // 2, (CELL - 8 - im.height) // 2))
                sheet.paste(bg, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "true_art_style_3_7_8_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
