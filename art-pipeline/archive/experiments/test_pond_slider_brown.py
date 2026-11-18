"""Test Pond Slider with 'warm brown oval carapace' (no 'olive' token that CLIP maps to green)
at s=0.68 and s=0.70 on animl_style_v5 (Step 1000 and Step 1250), plus check Candidate 12 from pond_slider candidates.
"""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
TEST_DIR = Path(__file__).parent / "batch_run_v5_ps_brown"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity/brain/cf192390-434f-49ae-a401-59243749f021")

NEG_BASE = (
    "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
    "heavy black outline, sticker, cross-hatching, pen hatching, "
    "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, extra limbs, tiling, text, watermark, border, cropped body, cut off, "
    "box turtle, speckled background, confetti"
)
STYLE_CLAUSE = (
    "rich natural colors, crisp distinct painted color steps, flat stylized "
    "gouache texture marks, clean thin dark charcoal-umber outline, "
    "isolated on a plain solid white background"
)


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = TEST_DIR / "inputs"
    out_dir = TEST_DIR / "outputs"
    in_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    # Copy the trimmed Pond Slider cutout from V5_EVAL
    src_cut = V5_EVAL / "inputs" / "08_pond_slider.png"
    im = Image.open(src_cut).convert("RGB")
    im.save(in_dir / "00_pond_slider.png")

    prompt = (
        f"animl_gouache_style, a pond slider, full body wildlife illustration, single animal, "
        f"warm brown oval carapace with thin yellow lines on scutes, bright red stripe behind eye, "
        f"dark forest-green head and legs with fine yellow stripes, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
    )
    pfile = TEST_DIR / "prompts.txt"
    pfile.write_text(prompt, encoding="utf-8")

    runs = [
        ("v5_s1000_068", LORA / "output" / "animl_style_v5-step00001000.safetensors", "0.68"),
        ("v5_s1000_071", LORA / "output" / "animl_style_v5-step00001000.safetensors", "0.71"),
        ("v5_s1250_068", LORA / "output" / "animl_style_v5-step00001250.safetensors", "0.68"),
        ("v5_s1250_071", LORA / "output" / "animl_style_v5-step00001250.safetensors", "0.71"),
    ]

    for tag, ckpt, s in runs:
        sub = out_dir / tag
        sub.mkdir(exist_ok=True)
        subprocess.run([
            str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
            "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
            "--outdir", str(sub),
            "--network_module", "networks.lora",
            "--network_weights", str(ckpt),
            "--network_mul", "1.0",
            "--max_embeddings_multiples", "3",
            "--from_file", str(pfile),
            "--image_path", str(in_dir),
            "--use_original_file_name",
            "--strength", s,
            "--W", "1024", "--H", "1024",
            "--steps", "28", "--sampler", "euler_a",
            "--scale", "7.0", "--bf16", "--sdpa",
            "--batch_size", "1", "--images_per_prompt", "1",
        ], check=True)

    cw, ch = 360, 360
    sheet = Image.new("RGB", (cw * 4, ch + 50), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arialbd.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for ci, (tag, _, s) in enumerate(runs):
        d.text((ci * cw + 12, 14), f"{tag} (s={s})", fill=(20, 20, 20), font=font)
        fp = out_dir / tag / "00_pond_slider.png"
        im_out = Image.open(fp).convert("RGBA")
        cut = remove(im_out)
        box = subject_bbox(cut)
        subj = cut.crop(box)
        scale = int((cw - 16) * 0.85) / max(subj.size)
        nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
        subj = subj.resize((nw, nh), Image.LANCZOS)
        cell = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        cell.paste(subj, ((cw - nw) // 2, (ch - nh) // 2), subj)
        sheet.paste(cell.convert("RGB"), (ci * cw, 48))
        d.rectangle([ci * cw, 48, (ci + 1) * cw - 1, 48 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "pond_slider_brown_v5_test.jpg", quality=92)
    print("Saved pond_slider_brown_v5_test.jpg", flush=True)


if __name__ == "__main__":
    main()
