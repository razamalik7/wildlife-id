"""Controlled single-variable test at EXACT same checkpoint (Step 1250), multiplier (1.0),
denoise strength (0.76), and input images (WORK / 'inputs') as Column 3!

Tests ONLY:
1. Dropping `pose_and_view` from Gemma completely (let the image cutout dictate pose!)
2. Keeping ONLY a concise 8-10 word `[color] [body part]` palette/marking hint from Gemma
   so it stops Box Turtle / Wood Duck species bleed WITHOUT diluting the prompt or fighting the pose/completion!
"""

import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
CTRL_DIR = WORK / "controlled_s1250_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

NEG_BASE = (
    "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
    "heavy black outline, sticker, cross-hatching, pen hatching, "
    "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, tiling, text, watermark, border, cropped body, cut off, "
    "speckled background, confetti"
)

STYLE_CLAUSE = (
    "rich natural colors, crisp distinct painted color steps, flat stylized "
    "gouache texture marks, clean thin dark charcoal-umber outline, "
    "isolated on a plain solid white background"
)

# Exact same 5 test species + Common Watersnake (6 total)
# Using the EXACT same input files from WORK / "inputs" that produced Column 3!
SPECIES = [
    ("04_mule_deer", "mule_deer", "Mule Deer",
     # Old Col 3 prompt:
     f"a mule deer, full body wildlife illustration, single deer standing with all four legs and clean hooves visible, three-quarter view, {STYLE_CLAUSE}",
     # New: NO pose override from Gemma, ONLY concise color/marking anchor + exact same completion clause:
     f"a mule deer with tan-brown coat and white rump patch, full body wildlife illustration, single deer standing with all four legs and clean hooves visible, {STYLE_CLAUSE}"),

    ("08_pond_slider", "pond_slider", "Pond Slider",
     f"a pond slider, full body wildlife illustration, single animal in swimming pose with all limbs and tail visible, three-quarter view, {STYLE_CLAUSE}",
     f"a pond slider turtle with plain dark olive-brown shell and red ear stripe, full body wildlife illustration, single animal in swimming pose with all limbs and tail visible, {STYLE_CLAUSE} --n {NEG_BASE}, yellow spots on shell, box turtle"),

    ("03_green-winged_teal", "green-winged_teal", "Green-winged Teal",
     f"a green-winged teal, full body wildlife illustration, single bird in floating pose with tucked feet, side three-quarter view, {STYLE_CLAUSE}",
     f"a green-winged teal with chestnut head, green eye-patch, speckled buff breast and grey flanks, full body wildlife illustration, single bird with webbed feet visible, {STYLE_CLAUSE}"),

    ("07_american_pika", "american_pika", "American Pika",
     f"an american pika, full body wildlife illustration, single small mammal with complete rounded body and all four paws visible, three-quarter view, {STYLE_CLAUSE}",
     f"an american pika with soft grizzled grey-buff fur and rounded ears, full body wildlife illustration, single small mammal with complete rounded body and all four paws visible, {STYLE_CLAUSE}"),

    ("06_california_ground_squirrel", "california_ground_squirrel", "California Ground Squirrel",
     f"a california ground squirrel, full body wildlife illustration, single animal with complete body and paws visible, three-quarter view, {STYLE_CLAUSE}",
     f"a california ground squirrel with mottled brown back, grey shoulder cape and buffy belly, full body wildlife illustration, single animal with complete body and paws visible, {STYLE_CLAUSE}"),

    ("09_common_watersnake", "common_watersnake", "Common Watersnake",
     f"a common watersnake, full body wildlife illustration, single snake with complete coiled body visible, three-quarter view, {STYLE_CLAUSE}",
     f"a common watersnake with dark slate-brown scales and reddish-brown crossbands, full body wildlife illustration, single snake with complete coiled body visible, {STYLE_CLAUSE}"),
]


def clean_and_center(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    im = Image.open(src).convert("RGBA")
    cut = remove(im)
    box = subject_bbox(cut)
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    canvas.convert("RGB").save(dst)


def main() -> None:
    CTRL_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = CTRL_DIR / "inputs"
    in_dir.mkdir(exist_ok=True)

    prompts = []
    for i, (inp_stem, slug, _, _, new_p) in enumerate(SPECIES):
        shutil.copyfile(WORK / "inputs" / f"{inp_stem}.png", in_dir / f"{i:02d}_{slug}.png")
        if "--n " not in new_p:
            prompts.append(f"{new_p} --n {NEG_BASE} --d 42")
        else:
            prompts.append(f"{new_p} --d 42")

    pfile = CTRL_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    # Test at Step 1250, mul 1.0, str 0.76 AND str 0.82 (EXACT same LoRA checkpoint and settings as Col 3!)
    for st in ["0.76", "0.82"]:
        odir = CTRL_DIR / f"out_{st}"
        cdir = CTRL_DIR / f"clean_{st}"
        odir.mkdir(exist_ok=True)
        cdir.mkdir(exist_ok=True)
        if len(list(odir.glob("*.png"))) < len(SPECIES):
            subprocess.run([
                str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
                "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
                "--outdir", str(odir),
                "--network_module", "networks.lora",
                "--network_weights", str(LORA / "output" / "animl_style_v4-step00001250.safetensors"),
                "--network_mul", "1.0",
                "--max_embeddings_multiples", "2",
                "--from_file", str(pfile),
                "--image_path", str(in_dir),
                "--use_original_file_name",
                "--strength", st,
                "--W", "1024", "--H", "1024",
                "--steps", "28", "--sampler", "euler_a",
                "--scale", "7.0", "--bf16", "--sdpa",
                "--batch_size", "1", "--images_per_prompt", "1",
            ], check=True)
        for p in odir.glob("*.png"):
            clean_and_center(p, cdir / p.name)

    # Assemble direct comparison sheet:
    # 1. Cutout Input | 2. Old Col 3 (s1250 @ 0.76, No Color Hint) | 3. Controlled s1250 @ 0.76 (Short Color Hint, No Pose) | 4. Controlled s1250 @ 0.82 (Short Color Hint, No Pose)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
        font_sm = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    cols = [
        "1. Input Cutout",
        "2. Old s1250 @ 0.76 (No Color Hint)",
        "3. s1250 @ 0.76 + Short Color Hint (No Pose)",
        "4. s1250 @ 0.82 + Short Color Hint (No Pose)",
    ]
    cw, ch = 360, 360
    W = len(cols) * cw
    H = 48 + len(SPECIES) * (ch + 32)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 12, 12), t, fill=(20, 20, 20), font=font)

    for ri, (inp_stem, slug, common, _, _) in enumerate(SPECIES):
        y_top = 48 + ri * (ch + 32)
        d.rectangle([0, y_top, W, y_top + 32], fill=(232, 232, 236))
        d.text((14, y_top + 6), f"{common} ({slug}) — Controlled Test: Exact Same s1250 LoRA & Input, Adding ONLY Concise Color Hint (No Pose)", fill=(10, 10, 10), font=font_sm)
        files = [
            WORK / "inputs" / f"{inp_stem}.png",
            WORK / "clean_str_0.76" / f"{inp_stem}.png",
            CTRL_DIR / "clean_0.76" / f"{ri:02d}_{slug}.png",
            CTRL_DIR / "clean_0.82" / f"{ri:02d}_{slug}.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 32))
            d.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "controlled_s1250_no_pose_compare.jpg", quality=92)
    print("Saved controlled_s1250_no_pose_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
