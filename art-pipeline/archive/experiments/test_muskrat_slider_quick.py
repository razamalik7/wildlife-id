"""Quick 35-second verification of the Muskrat & Pond Slider fixes before launching v6 training:
1. Muskrat:
   - Remove the word 'paws' from the positive prompt (prompting 'paws' made SDXL turn the jagged wet fur spikes along the lower belly at x=300..500 into extra paws/toes!).
   - Smooth the jagged wet fur spikes on the lower belly/flank (x > 280, y > 560) and curve the lower belly edge cleanly so only the real front paw under the chin has toe detail.
2. Pond Slider (120834277):
   - Fix the top shell bark-knob coordinate (the knob is at x=400..445, y < 365; do NOT cut into the carapace at x=450..560!).
   - Remove the `:1.35` prompt weight on red stripe (which bled red onto the neck and front leg!) while keeping the widened crimson red postorbital spot on the head in the input cutout.
"""

import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, percentile_filter
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
PREV_1AB_DIR = Path(__file__).parent / "batch_run_1ab_2_test"
TEST_DIR = Path(__file__).parent / "batch_run_ms_quick"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

NEG_CALM = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy shell, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, oil painting, blurry fur, stark white fur streaks, harsh contrast, "
    "heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, two tails, extra limbs, extra feet, extra paws, row of feet, claws on belly, fused legs, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_MARTEN_CALM = (
    "clean stylized 2d wildlife illustration, matte natural color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink fur and contour lines, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, "
    "isolated on a plain solid white background"
)


def make_clean_muskrat(src_path: Path, dst_path: Path) -> None:
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)
    yy, xx = np.ogrid[:1024, :1024]

    # 1. Trim the dangling wet root under the front paw (x < 305, y > 688) and smooth the lower belly edge (x=275..520)
    # so there are no protruding belly-fur lobes behind x=275 that look like extra feet!
    smooth_belly_y = np.where(
        xx < 265,
        688.0,
        688.0 - 24.0 * np.sin(np.clip((xx - 265.0) / 240.0, 0.0, 1.0) * np.pi),
    )
    cut_mask = (xx < 505) & (yy > smooth_belly_y)
    fg = fg & (~cut_mask)
    arr[cut_mask] = 255.0

    # 2. In the lower belly/flank region behind the cheek (x > 265, y > 540), damp the spiky wet fur clumps
    # so SDXL doesn't see dark-and-light toe-like spikes along the bottom edge!
    med_flank = np.stack([percentile_filter(arr[:, :, c], percentile=45, size=15) for c in range(3)], axis=-1)
    smooth_flank = np.stack([gaussian_filter(med_flank[:, :, c], sigma=4.0) for c in range(3)], axis=-1)

    # Blend smoothly into the lower belly/flank (x > 260, y > 550)
    w_x = np.clip((xx - 255.0) / 45.0, 0.0, 1.0)
    w_y = np.clip((yy - 540.0) / 55.0, 0.0, 1.0)
    w_flank = (w_x * w_y)[:, :, None]

    out = arr * (1.0 - 0.85 * w_flank) + smooth_flank * (0.85 * w_flank)
    out_rgb = np.where(fg[:, :, None], np.clip(out, 0, 255).astype(np.uint8), 255)
    Image.fromarray(out_rgb).save(dst_path)


def make_clean_slider(src_path: Path, dst_path: Path) -> None:
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    fg = np.any(arr < 250, axis=-1)
    yy, xx = np.ogrid[:1024, :1024]

    # Find exact coordinates of the little bump on top of the shell:
    # Let's check where the upper carapace contour has a small protrusion around x=380..460, y < 385
    # A clean straight/convex tangent across the upper-left carapace from (x=360, y=405) to (x=470, y=348):
    tangent_y = 405.0 - (405.0 - 348.0) * (xx - 360.0) / (470.0 - 360.0)
    is_bump = (xx >= 360) & (xx <= 470) & (yy < tangent_y)
    arr[is_bump] = 255.0
    fg = fg & (~is_bump)

    # Widen and brighten ONLY the red postorbital ear patch on the head (xx > 655, yy < 305)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_head_red = (xx > 650) & (xx < 745) & (yy > 220) & (yy < 310) & (r > g + 6.0) & fg
    red_mask_float = gaussian_filter(is_head_red.astype(np.float32), sigma=2.8)
    is_expanded_red = (red_mask_float > 0.16) & fg & (xx > 655) & (yy < 305)

    arr[is_expanded_red, 0] = np.clip(arr[is_expanded_red, 0] * 1.45 + 35.0, 180.0, 230.0)
    arr[is_expanded_red, 1] = np.clip(arr[is_expanded_red, 1] * 0.40, 18.0, 50.0)
    arr[is_expanded_red, 2] = np.clip(arr[is_expanded_red, 2] * 0.40, 18.0, 50.0)

    # Also gently suppress the pale bluish sky glare on the upper carapace (xx < 600, sat < 0.25, lum > 105)
    # without blurring the scute pattern!
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / np.maximum(max_c, 1.0)
    is_shell_glare = (xx < 610) & (yy > 340) & (yy < 650) & fg
    glare_w = np.where(
        is_shell_glare,
        np.clip((lum - 105.0) / 55.0, 0.0, 1.0) * np.clip((0.28 - sat) / 0.18, 0.0, 1.0),
        0.0,
    )
    # Tint sky glare toward warm olive-brown while keeping local scute contrast
    arr[:, :, 0] = arr[:, :, 0] * (1.0 - 0.30 * glare_w)
    arr[:, :, 1] = arr[:, :, 1] * (1.0 - 0.35 * glare_w)
    arr[:, :, 2] = arr[:, :, 2] * (1.0 - 0.55 * glare_w)

    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(dst_path)


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, lora_path: Path, strength: str, cfg: str, mul: str) -> None:
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= 2:
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
    in_dir = TEST_DIR / "inputs"
    in_dir.mkdir(exist_ok=True)

    make_clean_muskrat(PREV_1AB_DIR / "inputs_raw" / "01_muskrat.png", in_dir / "00_muskrat.png")
    make_clean_slider(PREV_1AB_DIR / "inputs_slider" / "01_slider_120834277.png", in_dir / "01_slider.png")

    prompts = [
        # 00: Muskrat (NO 'paws' or 'feet' in prompt!)
        f"animl_gouache_style, a muskrat, full body wildlife illustration, single animal crouching in profile with rounded body, dark eyes, dark muzzle, uniform warm brown coat, darker brown back, lighter tan cheeks, short brown whiskers, {STYLE_MARTEN_CALM} --n {NEG_CALM} --d 42",
        # 01: Pond Slider (NO ':1.35' red bleed onto legs! Unweighted red stripe behind eye + green/yellow neck & legs)
        f"animl_gouache_style, a pond slider, full body wildlife illustration, single animal standing with webbed clawed feet visible, smooth dark olive-brown oval carapace with dark brown scutes, dark olive-green head and legs with bright yellow longitudinal stripes, distinct red postorbital stripe behind the eye, {STYLE_MARTEN_CALM} --n {NEG_CALM}, yellow shell, box turtle shell, red legs, red neck --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1000 = LORA / "output" / "animl_style_v5-step00001000.safetensors"
    lora_1250 = LORA / "output" / "animl_style_v5-step00001250.safetensors"

    run_sdxl(in_dir, pf, TEST_DIR / "col2_s074_step1000", lora_1000, "0.74", "5.5", "1.0")
    run_sdxl(in_dir, pf, TEST_DIR / "col3_s076_step1250", lora_1250, "0.76", "5.5", "1.0")
    run_sdxl(in_dir, pf, TEST_DIR / "col4_s079_step1250", lora_1250, "0.79", "5.5", "1.0")

    CELL = 420
    HEADER_H = 52
    ROW_LABEL_H = 34
    W = 4 * CELL
    H = HEADER_H + 2 * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 17)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. Fixed Input",
        "2. step1000 (s=0.74)",
        "3. step1250 (s=0.76)",
        "4. step1250 (s=0.79)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 15), title, fill=(20, 20, 30), font=font_b)

    in_imgs = sorted(in_dir.glob("*.png"))
    c2_imgs = sorted((TEST_DIR / "col2_s074_step1000").glob("*.png"))
    c3_imgs = sorted((TEST_DIR / "col3_s076_step1250").glob("*.png"))
    c4_imgs = sorted((TEST_DIR / "col4_s079_step1250").glob("*.png"))

    for r_idx, lbl in enumerate(["Muskrat (Belly Spikes Smoothed + No 'paws' in Prompt)", "Pond Slider (Clean Shell Tangent + Head-Only Red Stripe)"]):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        for c_idx, im in enumerate([Image.open(in_imgs[r_idx]).convert("RGB"), clean_post(c2_imgs[r_idx]), clean_post(c3_imgs[r_idx]), clean_post(c4_imgs[r_idx])]):
            thumb = im.resize((CELL - 8, CELL - 8), Image.LANCZOS)
            sheet.paste(thumb, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "ms_quick_fix_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}")


if __name__ == "__main__":
    main()
