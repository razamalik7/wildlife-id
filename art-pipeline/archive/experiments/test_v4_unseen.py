"""Fast batch test of animl_style_v4 on the 11 unseen species in picks_overnight.json.

Uses sdxl_gen_img.py's directory + prompt-file batching so the 6.5GB SDXL base
checkpoint and LoRA weights are loaded into VRAM only once per strength pass!
Also tests a 'completed' cutout variant for grass/water-clipped species.
"""

import json
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
BATCH_OLD = Path(__file__).parent / "batch_run"
WORK = Path(__file__).parent / "batch_run_v4"

UNSEEN_SPECIES = [
    ("northern_shoveler", "Northern Shoveler", "floating"),
    ("northern_pintail", "Northern Pintail", "floating"),
    ("american_wigeon", "American Wigeon", "floating"),
    ("green-winged_teal", "Green-winged Teal", "floating"),
    ("mule_deer", "Mule Deer", "grass_hooves"),
    ("muskrat", "Muskrat", "swimming"),
    ("california_ground_squirrel", "California Ground Squirrel", "standing"),
    ("american_pika", "American Pika", "rock_perch"),
    ("pond_slider", "Pond Slider", "swimming"),
    ("common_watersnake", "Common Watersnake", "coiled"),
    ("gopher_tortoise", "Gopher Tortoise", "walking"),
]

NEG_BASE = (
    "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
    "heavy black outline, sticker, cross-hatching, pen hatching, "
    "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_CLAUSE = (
    "rich natural colors, crisp distinct painted color steps, flat stylized "
    "gouache texture marks, clean thin dark charcoal-umber outline, "
    "isolated on a plain solid white background"
)

POSE_CLAUSE = {
    "floating": "full body wildlife illustration, single bird in floating pose with tucked feet, side three-quarter view",
    "grass_hooves": "full body wildlife illustration, single deer standing with all four legs and clean hooves visible, three-quarter view",
    "swimming": "full body wildlife illustration, single animal in swimming pose with all limbs and tail visible, three-quarter view",
    "standing": "full body wildlife illustration, single animal with complete body and paws visible, three-quarter view",
    "rock_perch": "full body wildlife illustration, single small mammal with complete rounded body and all four paws visible, three-quarter view",
    "coiled": "full body wildlife illustration, single snake with complete coiled body visible, three-quarter view",
    "walking": "full body wildlife illustration, single tortoise walking with shell and all four legs visible, three-quarter view",
}


def make_soft_edge_cutout(cut_p: Path, dst_p: Path, mode: str) -> None:
    """Prepare the cutout input image.
    For clipped bottom edges (grass_hooves, swimming, rock_perch), feather/extend
    the bottom edge slightly and shift up so SDXL img2img has room & latent signal
    to complete feet/hooves/paws instead of hitting a razor-sharp white wall.
    """
    im = Image.open(cut_p).convert("RGB")
    if mode in ("grass_hooves", "rock_perch", "swimming", "floating"):
        # Find non-white bounding box
        import numpy as np
        arr = np.array(im)
        mask = (arr < 248).any(axis=2)
        ys, xs = np.where(mask)
        if len(ys) > 0:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            # Feather the bottom 28 pixels of the cutout slightly downward so latent noise blends
            crop = im.crop((x0, y0, x1 + 1, y1 + 1))
            cw, ch = crop.size
            # Scale slightly smaller (68% instead of 76%) and shift up by 40px so lower limbs have room
            scale = int(1024 * 0.68) / max(cw, ch)
            nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
            crop_resized = crop.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGB", (1024, 1024), (255, 255, 255))
            px = (1024 - nw) // 2
            py = max(40, (1024 - nh) // 2 - 45)
            canvas.paste(crop_resized, (px, py))
            # Softly smudge/extrude the bottom 18px strip downward by 36px at 35% opacity
            # to seed non-white latent variance where hooves/paddling feet belong
            if mode in ("grass_hooves", "swimming", "rock_perch"):
                strip_h = min(24, nh // 6)
                bottom_strip = crop_resized.crop((0, nh - strip_h, nw, nh))
                ext_h = 48 if mode != "grass_hooves" else 64
                bottom_ext = bottom_strip.resize((nw, ext_h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(8))
                white_bg = Image.new("RGB", (nw, ext_h), (255, 255, 255))
                seeded = Image.blend(white_bg, bottom_ext, 0.42)
                canvas.paste(seeded, (px, py + nh - 6))
            canvas.save(dst_p)
            return
    shutil.copyfile(cut_p, dst_p)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    in_dir = WORK / "inputs"
    in_dir.mkdir(exist_ok=True)

    # Also copy animl_style_v4-step00001250.safetensors to animl_style_v4.safetensors
    step1250 = LORA / "output" / "animl_style_v4-step00001250.safetensors"
    default_lora = LORA / "output" / "animl_style_v4.safetensors"
    if step1250.exists():
        shutil.copyfile(step1250, default_lora)

    prompts = []
    for idx, (slug, common, mode) in enumerate(UNSEEN_SPECIES):
        src_photo = BATCH_OLD / f"{slug}_photo.png"
        src_cut = BATCH_OLD / f"{slug}_cutout.png"
        shutil.copyfile(src_photo, WORK / f"{slug}_photo.png")
        shutil.copyfile(src_cut, WORK / f"{slug}_cutout.png")

        # Named 00_slug.png so sorted order is deterministic
        inp_p = in_dir / f"{idx:02d}_{slug}.png"
        make_soft_edge_cutout(src_cut, inp_p, mode)

        article = "an" if common[0].lower() in "aeiou" else "a"
        pose = POSE_CLAUSE[mode]
        p = f"{article} {common.lower()}, {pose}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        prompts.append(p)

    prompt_file = WORK / "prompts.txt"
    prompt_file.write_text("\n".join(prompts), encoding="utf-8")

    strengths = [0.68, 0.76, 0.84]
    for s in strengths:
        outdir = WORK / f"out_str_{s}"
        outdir.mkdir(exist_ok=True)
        existing = list(outdir.glob("*.png"))
        if len(existing) >= len(UNSEEN_SPECIES):
            print(f"Strength {s} already complete, skipping")
            continue
        print(f"Running strength {s} across all 11 species...", flush=True)
        cmd = [
            str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
            "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
            "--outdir", str(outdir),
            "--network_module", "networks.lora",
            "--network_weights", str(default_lora),
            "--network_mul", "1.0",
            "--from_file", str(prompt_file),
            "--image_path", str(in_dir),
            "--use_original_file_name",
            "--strength", str(s),
            "--W", "1024", "--H", "1024",
            "--steps", "28", "--sampler", "euler_a",
            "--scale", "7.0", "--bf16", "--sdpa",
            "--batch_size", "1", "--images_per_prompt", "1",
        ]
        subprocess.run(cmd, check=True)

    # Build composite contact sheets
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    # Split into 2 sheets so images are large and crisp:
    # Sheet A: 6 Mammals/Reptiles (Mule Deer, Muskrat, Ground Squirrel, Pika, Pond Slider, Watersnake, Gopher Tortoise -> 7 non-duck species)
    # Sheet B: 4 Waterfowl (Northern Shoveler, Northern Pintail, American Wigeon, Green-winged Teal)
    groups = [
        ("v4_unseen_land_and_semi_aquatic.jpg", [x for x in UNSEEN_SPECIES if x[2] != "floating"]),
        ("v4_unseen_waterfowl.jpg", [x for x in UNSEEN_SPECIES if x[2] == "floating"]),
    ]

    artifact_dir = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")
    for fname, items in groups:
        cols = ["Reference Photo", "Cutout Input", "v4 LoRA @ 0.68", "v4 LoRA @ 0.76", "v4 LoRA @ 0.84"]
        cell_w, cell_h = 340, 340
        header_h = 48
        row_label_h = 32
        W = len(cols) * cell_w
        H = header_h + len(items) * (cell_h + row_label_h)
        sheet_im = Image.new("RGB", (W, H), (248, 248, 248))
        draw = ImageDraw.Draw(sheet_im)

        for ci, col_title in enumerate(cols):
            draw.text((ci * cell_w + 14, 12), col_title, fill=(20, 20, 20), font=font)

        for ri, (slug, common, mode) in enumerate(items):
            idx = [x[0] for x in UNSEEN_SPECIES].index(slug)
            y_top = header_h + ri * (cell_h + row_label_h)
            draw.rectangle([0, y_top, W, y_top + row_label_h], fill=(232, 232, 236))
            draw.text((14, y_top + 5), f"{common} ({slug}) — mode: {mode}", fill=(10, 10, 10), font=font_sm)

            row_files = [
                WORK / f"{slug}_photo.png",
                in_dir / f"{idx:02d}_{slug}.png",
                WORK / "out_str_0.68" / f"{idx:02d}_{slug}.png",
                WORK / "out_str_0.76" / f"{idx:02d}_{slug}.png",
                WORK / "out_str_0.84" / f"{idx:02d}_{slug}.png",
            ]
            for ci, fp in enumerate(row_files):
                cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
                if fp.exists():
                    im = Image.open(fp).convert("RGB")
                    im.thumbnail((cell_w - 8, cell_h - 8), Image.LANCZOS)
                    cell.paste(im, ((cell_w - im.width) // 2, (cell_h - im.height) // 2))
                sheet_im.paste(cell, (ci * cell_w, y_top + row_label_h))
                draw.rectangle(
                    [ci * cell_w, y_top + row_label_h, (ci + 1) * cell_w - 1, y_top + row_label_h + cell_h - 1],
                    outline=(210, 210, 210),
                )

        sheet_im.save(WORK / fname, quality=92)
        sheet_im.save(artifact_dir / fname, quality=92)
        print(f"Saved {fname}", flush=True)


if __name__ == "__main__":
    main()
