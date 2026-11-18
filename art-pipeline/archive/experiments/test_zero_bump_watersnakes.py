"""Pose-Locked + Surface-Only LoRA Pipeline for All 8 Common Watersnake Candidates.

Fixes both root causes discovered in earlier runs:
1. True Block-Filtered Surface LoRA (`animl_style_v6_surface_only.safetensors`):
   - `networks/lora.py`'s `create_network_from_weights` ignores `block_lr_weight` during inference.
   - By zeroing out the 612 32x32 U-Net bottleneck matrices (`input_blocks_7..8`, `middle_block`, `output_blocks_0..2`)
     directly in `animl_style_v6_surface_only.safetensors` while keeping 100% of the 64x64 and 128x128 surface style
     blocks (`input_blocks_1..3`, `output_blocks_3..8`), the LoRA's memorized "head-on-the-right coiled snake" pose
     bias is physically zeroed out!
2. Zero-Creature-Noun Surface Prompts + Head-Preserving Spatial Blend:
   - Removing creature nouns (`"animal"`, `"snake"`, `"serpent"`, `"head"`) from the global prompt prevents SDXL Base's
     spatial cross-attention from ever painting fake eyes/snouts onto U-bends (`Options 4, 5, 6`) or toes onto moss
     notches (`Option 2`).
   - Blending `s=0.35` over a soft Gaussian window around each photo's true head with `s=0.46` across the body coils
     preserves the exact round eye, labial bars, and jaw structure while rendering the full body in crisp matte gouache.
"""

import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
ALL8_DIR = Path(__file__).parent / "batch_run_all8_snakes"
TEST_DIR = Path(__file__).parent / "batch_run_all8_snakes_pose_locked_v2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SNAKE_8_CANDS = [
    ("00_opt1_222925393", "222925393", "Option 1 (222925393): S-Coil on Rock", (740, 225, 115)),
    ("01_opt2_428226899", "428226899", "Option 2 (428226899): Side-Profile S-Loop", (205, 435, 125)),
    ("02_opt3_13608867",  "13608867",  "Option 3 (13608867): Layered Resting Coil", (600, 415, 105)),
    ("03_opt4_284103518", "284103518", "Option 4 (284103518): Flat S-Curve on Ground", (465, 565, 130)),
    ("04_opt5_20386464",  "20386464",  "Option 5 (20386464): S-Coil on Flat Stone", (505, 485, 110)),
    ("05_opt6_8478653",   "8478653",   "Option 6 (8478653): Tight Spiral Coil", (515, 485, 110)),
    ("06_opt7_526919366", "526919366", "Option 7 (526919366): Basking S-Coil", (825, 595, 110)),
    ("07_opt8_13810420",  "13810420",  "Option 8 (13810420): Raised-Head S-Curve", (425, 485, 100)),
]

STYLE_CLAUSE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_CLEAN = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy, shiny, specular highlight, plastic, airbrushed, "
    "watercolor, painterly brush swirls, oil painting, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, moss, root, mud, water, pond, "
    "grass, dirt, gravel, ground plane, cast shadow, drop shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, double headed, extra head, extra eye, two tails, extra limbs, claws, toes, feet, legs, lump, bump, "
    "white stripes, stark white bands, high contrast white rings, "
    "tiling, text, watermark, border, cropped body, cut off"
)


def get_aligned_raw(raw_photo: Path) -> np.ndarray:
    raw_im = Image.open(raw_photo).convert("RGBA")
    cut = remove(raw_im)
    arr = np.array(cut)
    arr[arr[:, :, 3] < 140, 3] = 0
    box = subject_bbox(Image.fromarray(arr))
    raw_crop = raw_im.crop(box)
    scale = int(1024 * 0.78) / max(raw_crop.size)
    nw, nh = max(1, round(raw_crop.width * scale)), max(1, round(raw_crop.height * scale))
    raw_res = raw_crop.resize((nw, nh), Image.LANCZOS)
    raw_canvas = Image.new("RGB", (1024, 1024), (255, 255, 255))
    raw_canvas.paste(raw_res.convert("RGB"), ((1024 - nw) // 2, (1024 - nh) // 2))
    return np.array(raw_canvas)


def prepare_clean_input(stem: str, src_cutout: Path, raw_photo: Path, dst_path: Path) -> None:
    im = Image.open(src_cutout).convert("RGB")
    arr = np.array(im, dtype=np.uint8)

    if stem == "01_opt2_428226899":
        # Smooth out the moss-bitten sawtooth spikes along the lower-right belly edge
        pil_edit = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil_edit)
        draw.polygon([
            (465, 622), (565, 584), (665, 548), (765, 514), (860, 482), (905, 448),
            (925, 760), (465, 760)
        ], fill=(255, 255, 255))
        arr = np.array(pil_edit)

    elif stem == "03_opt4_284103518":
        raw_np = get_aligned_raw(raw_photo)
        # 1. Restore missing inner-right loop bite at x=765..832, y=490..548
        mask4 = Image.new("L", (1024, 1024), 0)
        ImageDraw.Draw(mask4).polygon([(765, 492), (832, 490), (818, 544), (765, 548)], fill=255)
        m4 = np.array(mask4) > 128
        arr[m4] = raw_np[m4]
        # 2. Smoothly trim the outer-left tail-tip/beetle bump at x=130..204, y=415..492
        pil_edit = Image.fromarray(arr)
        ImageDraw.Draw(pil_edit).polygon([(130, 415), (204, 422), (168, 468), (144, 492), (130, 492)], fill=(255, 255, 255))
        arr = np.array(pil_edit)

    elif stem == "04_opt5_20386464":
        hole = np.all(arr[315:355, 698:725] > 240, axis=-1)
        if np.any(hole):
            arr[315:355, 698:725][hole] = arr[315:355, 680:707][hole]

    elif stem == "06_opt7_526919366":
        pil_edit = Image.fromarray(arr)
        ImageDraw.Draw(pil_edit).polygon([(680, 650), (760, 662), (835, 660), (878, 642), (910, 770), (680, 770)], fill=(255, 255, 255))
        arr = np.array(pil_edit)

    elif stem == "07_opt8_13810420":
        raw_np = get_aligned_raw(raw_photo)
        # 1. Restore shadowed left edge of snout at x=402..432, y=458..532
        mask8 = Image.new("L", (1024, 1024), 0)
        ImageDraw.Draw(mask8).polygon([(412, 456), (432, 454), (432, 530), (412, 532), (402, 505), (404, 475)], fill=255)
        m8 = np.array(mask8) > 128
        arr[m8] = np.clip(raw_np[m8].astype(np.float32) * 1.18 + 8, 0, 210).astype(np.uint8)
        # 2. Trim the gravel cast-shadow bulge on the outer right curve (x=865..905, y=435..570)
        pil_edit = Image.fromarray(arr)
        ImageDraw.Draw(pil_edit).polygon([(872, 435), (910, 435), (910, 575), (852, 575), (874, 525), (880, 478)], fill=(255, 255, 255))
        arr = np.array(pil_edit)

    Image.fromarray(arr).save(dst_path)


def clean_post(src_path: Path, dst_path: Path, ref_input_path: Path) -> Image.Image:
    """Clamps generated silhouette to the repaired input cutout (dilated by 2px for charcoal outline)
    without re-cropping or shifting coordinates, so head-blend alignment is 100% exact."""
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    im = Image.open(src_path).convert("RGBA")
    cut = remove(im)
    arr = np.array(cut)
    arr[arr[:, :, 3] < 165, 3] = 0

    ref_arr = np.array(Image.open(ref_input_path).convert("RGB"))
    ref_fg = np.any(ref_arr < 246, axis=-1)
    allowed_fg = binary_dilation(ref_fg, iterations=2)
    arr[~allowed_fg, 3] = 0

    # Composite directly onto pure white 1024x1024 canvas at exact same pixel coordinates
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb_arr = arr[:, :, :3].astype(np.float32)
    comp = (rgb_arr * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    rgb = Image.fromarray(comp)
    rgb.save(dst_path)
    return rgb


def run_sdxl_surface_only(
    in_dir: Path,
    prompt_file: Path,
    out_dir: Path,
    clean_dir: Path,
    lora_path: Path,
    strength: str,
    cfg: str = "4.5",
    mul: str = "0.95",
) -> None:
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
        clean_post(png, clean_dir / png.name, in_dir / png.name)


def make_head_preserved_blend(
    head_dir: Path,
    body_dir: Path,
    ref_in_dir: Path,
    blend_dir: Path,
) -> None:
    """Blends the sharp head pass (s=0.35) around each snake's true head coordinates with
    the stronger matte gouache body pass (s=0.46) across the coils."""
    blend_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:1024, 0:1024]
    for stem, _, _, (hx, hy, rad) in SNAKE_8_CANDS:
        p_head = head_dir / f"{stem}.png"
        p_body = body_dir / f"{stem}.png"
        p_ref = ref_in_dir / f"{stem}.png"
        if not (p_head.exists() and p_body.exists()):
            continue
        arr_h = np.array(Image.open(p_head).convert("RGB"), dtype=np.float32)
        arr_b = np.array(Image.open(p_body).convert("RGB"), dtype=np.float32)
        dist2 = (xx - hx) ** 2 + (yy - hy) ** 2
        w_head = np.exp(-dist2 / (2.0 * (rad ** 2)))[:, :, None]
        comp = (arr_h * w_head + arr_b * (1.0 - w_head)).clip(0, 255).astype(np.uint8)
        # Re-clamp silhouette cleanly
        ref_arr = np.array(Image.open(p_ref).convert("RGB"))
        allowed = binary_dilation(np.any(ref_arr < 246, axis=-1), iterations=2)
        comp[~allowed] = 255
        Image.fromarray(comp).save(blend_dir / f"{stem}.png")


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = TEST_DIR / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    for stem, pid, _, _ in SNAKE_8_CANDS:
        prepare_clean_input(
            stem,
            ALL8_DIR / "inputs" / f"{stem}.png",
            ALL8_DIR / "raw_photos" / f"{pid}.jpg",
            in_dir / f"{stem}.png",
        )

    # Zero-creature-noun surface style prompts:
    # By omitting "animal", "snake", "serpent", and "head", SDXL Base never attempts to paint
    # eyes/mouths onto body U-bends or toes onto belly edges!
    prompts = [
        # 00: 222925393
        f"animl_gouache_style, scientific illustration, matte gouache painting of warm tan and dark chocolate-brown patterned scales with reddish-brown markings, pale cream-tan underside scales, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 01: 428226899
        f"animl_gouache_style, scientific illustration, matte gouache painting of warm brown and rufous-tan patterned scales with dark brown markings, olive-brown scales, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 02: 13608867
        f"animl_gouache_style, scientific illustration, matte gouache painting of warm terracotta-orange and dark brown patterned scales, copper-orange and chestnut-brown scales, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 03: 284103518
        f"animl_gouache_style, scientific illustration, matte gouache painting of matte olive-tan and warm brown patterned scales with darker brown markings, pale cream-tan underside, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 04: 20386464
        f"animl_gouache_style, scientific illustration, matte gouache painting of pale silvery-tan and gray-brown patterned scales with soft brown markings, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 05: 8478653
        f"animl_gouache_style, scientific illustration, matte gouache painting of warm pinkish-tan and buff-orange patterned scales with soft brown markings, olive-tan scales, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 06: 526919366
        f"animl_gouache_style, scientific illustration, matte gouache painting of matte dark brown and warm tan patterned scales, pale yellow-tan underside scales, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
        # 07: 13810420
        f"animl_gouache_style, scientific illustration, matte gouache painting of matte slate-brown and warm tan patterned scales with subtle dark markings, {STYLE_CLAUSE} --n {NEG_CLEAN} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_surface = LORA / "output" / "animl_style_v6_surface_only.safetensors"

    run_sdxl_surface_only(in_dir, pf, TEST_DIR / "raw_s035", TEST_DIR / "clean_s035", lora_surface, "0.35", "4.5", "1.0")
    run_sdxl_surface_only(in_dir, pf, TEST_DIR / "raw_s046", TEST_DIR / "clean_s046", lora_surface, "0.46", "4.5", "1.0")
    run_sdxl_surface_only(in_dir, pf, TEST_DIR / "raw_s052", TEST_DIR / "clean_s052", lora_surface, "0.52", "4.5", "1.0")

    make_head_preserved_blend(
        TEST_DIR / "clean_s035",
        TEST_DIR / "clean_s046",
        in_dir,
        TEST_DIR / "clean_blend",
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
        "1. Repaired Cutout Input",
        "2. Surface-LoRA (s=0.35)",
        "3. Surface-LoRA (s=0.46)",
        "4. Surface-LoRA (s=0.52)",
        "5. Head-Locked Gouache Composite",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 10, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, pid, lbl, _) in enumerate(SNAKE_8_CANDS):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            in_dir / f"{stem}.png",
            TEST_DIR / "clean_s035" / f"{stem}.png",
            TEST_DIR / "clean_s046" / f"{stem}.png",
            TEST_DIR / "clean_s052" / f"{stem}.png",
            TEST_DIR / "clean_blend" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", (CELL - 8, CELL - 8), (255, 255, 255))
                bg.paste(im, ((CELL - 8 - im.width) // 2, (CELL - 8 - im.height) // 2))
                sheet.paste(bg, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_all8_watersnakes_surface_only_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
