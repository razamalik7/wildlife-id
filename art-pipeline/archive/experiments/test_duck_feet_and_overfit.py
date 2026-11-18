"""Test script for two key questions from the user:
1. Why did v4 overfit to Box Turtle / Gopher Snake (and get background speckle)?
   -> Compare checkpoint steps (500, 750, 1000, 1250) + explicit species color/marking prompts
      on Pond Slider, Common Watersnake, California Ground Squirrel, and Mule Deer.
2. What does floating duck foot completion (paddling vs tucked webbed feet) look like
   on Northern Pintail and American Wigeon when we seed/sketch webbed feet under the waterline?
"""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
TEST_DIR = WORK / "diag_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

NEG_BASE = (
    "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
    "heavy black outline, sticker, cross-hatching, pen hatching, "
    "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, tiling, text, watermark, border, cropped body, cut off, "
    "yellow spots on shell, box turtle, speckled background, confetti"
)

STYLE_CLAUSE = (
    "flat stylized gouache wildlife illustration, crisp distinct painted color steps, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)


def make_duck_foot_inputs(slug: str, src_cut: Path, dst_paddling: Path, dst_tucked: Path) -> None:
    """Create two input variants for a floating duck cutout:
    1. Paddling webbed feet painted beneath the waterline
    2. Tucked swimming webbed feet painted beneath the rear belly
    """
    im = Image.open(src_cut).convert("RGB")
    import numpy as np
    arr = np.array(im)
    mask = (arr < 245).any(axis=2)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = im.crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = crop.size

    # Scale so width is ~680px, leaving room below for feet
    scale = 680 / max(cw, ch)
    nw, nh = round(cw * scale), round(ch * scale)
    crop_r = crop.resize((nw, nh), Image.LANCZOS)

    # Foot color: slate-grey for Pintail/Wigeon (their real foot color!) or warm orange-tan
    foot_col = (115, 120, 128) if "pintail" in slug or "wigeon" in slug else (195, 125, 55)
    web_col = (140, 145, 152) if "pintail" in slug or "wigeon" in slug else (215, 145, 70)
    outline_col = (55, 50, 48)

    # 1. Paddling pose: two webbed feet dangling/paddling at ~45 deg under belly
    canv_pad = Image.new("RGB", (1024, 1024), (255, 255, 255))
    px = (1024 - nw) // 2
    py = (1024 - nh) // 2 - 45
    canv_pad.paste(crop_r, (px, py))
    d = ImageDraw.Draw(canv_pad)
    by = py + nh - 12  # waterline
    # Round the flat belly slightly first
    belly_col = crop_r.getpixel((nw // 2, nh - 18))
    d.chord([px + int(nw * 0.18), by - 28, px + int(nw * 0.78), by + 26], start=0, end=180, fill=belly_col)
    # Front paddling foot (kick back)
    fx1, fy1 = px + int(nw * 0.46), by + 10
    d.line([(fx1, fy1), (fx1 - 28, fy1 + 42)], fill=foot_col, width=10)
    d.polygon([(fx1 - 28, fy1 + 42), (fx1 - 75, fy1 + 58), (fx1 - 45, fy1 + 82), (fx1 - 16, fy1 + 65)], fill=web_col, outline=outline_col)
    # Rear paddling foot
    fx2, fy2 = px + int(nw * 0.32), by + 8
    d.line([(fx2, fy2), (fx2 - 36, fy2 + 36)], fill=foot_col, width=10)
    d.polygon([(fx2 - 36, fy2 + 36), (fx2 - 82, fy2 + 44), (fx2 - 58, fy2 + 72), (fx2 - 26, fy2 + 56)], fill=web_col, outline=outline_col)
    canv_pad = canv_pad.filter(ImageFilter.GaussianBlur(1.2))
    canv_pad.save(dst_paddling)

    # 2. Tucked pose: rounded belly + webbed feet tucked neatly backward under tail/flank
    canv_tuck = Image.new("RGB", (1024, 1024), (255, 255, 255))
    canv_tuck.paste(crop_r, (px, py))
    d2 = ImageDraw.Draw(canv_tuck)
    d2.chord([px + int(nw * 0.16), by - 32, px + int(nw * 0.80), by + 30], start=0, end=180, fill=belly_col)
    tx, ty = px + int(nw * 0.30), by + 12
    d2.polygon([(tx, ty), (tx - 65, ty + 14), (tx - 52, ty + 38), (tx + 5, ty + 22)], fill=web_col, outline=outline_col)
    d2.polygon([(tx + 28, ty + 6), (tx - 35, ty + 22), (tx - 22, ty + 44), (tx + 32, ty + 24)], fill=foot_col, outline=outline_col)
    canv_tuck = canv_tuck.filter(ImageFilter.GaussianBlur(1.2))
    canv_tuck.save(dst_tucked)


def clean_and_center(src: Path, dst: Path) -> None:
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
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    duck_in = TEST_DIR / "duck_inputs"
    duck_in.mkdir(exist_ok=True)

    # Part A: Build duck foot-completion inputs for Pintail and Wigeon
    ducks = [
        ("northern_pintail", "Northern Pintail", "grey bill, chocolate brown head, white neck stripe, grey body, grey webbed feet"),
        ("american_wigeon", "American Wigeon", "iridescent green eye band, cream forehead crown, pinkish-brown breast, grey webbed feet"),
    ]
    duck_prompts = []
    for idx, (slug, common, markings) in enumerate(ducks):
        src_cut = WORK / f"{slug}_cutout.png"
        pad_p = duck_in / f"{idx*2:02d}_{slug}_paddling.png"
        tuck_p = duck_in / f"{idx*2+1:02d}_{slug}_tucked.png"
        make_duck_foot_inputs(slug, src_cut, pad_p, tuck_p)
        p_pad = (
            f"a {common.lower()}, {markings}, full body floating duck with paddling webbed feet visible underneath, "
            f"{STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        )
        p_tuck = (
            f"a {common.lower()}, {markings}, full body floating duck with rounded belly and tucked webbed feet underneath, "
            f"{STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        )
        duck_prompts.extend([p_pad, p_tuck])

    duck_pfile = TEST_DIR / "duck_prompts.txt"
    duck_pfile.write_text("\n".join(duck_prompts), encoding="utf-8")

    # Run duck foot completion at Step 750 (60 epochs, less overfit) and Step 1250 @ strength 0.72
    duck_out = TEST_DIR / "duck_outputs"
    duck_out.mkdir(exist_ok=True)
    if len(list(duck_out.glob("*.png"))) < 4:
        cmd = [
            str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
            "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
            "--outdir", str(duck_out),
            "--network_module", "networks.lora",
            "--network_weights", str(LORA / "output" / "animl_style_v4-step00000750.safetensors"),
            "--network_mul", "0.85",
            "--from_file", str(duck_pfile),
            "--image_path", str(duck_in),
            "--use_original_file_name",
            "--strength", "0.70",
            "--W", "1024", "--H", "1024",
            "--steps", "28", "--sampler", "euler_a",
            "--scale", "7.0", "--bf16", "--sdpa",
            "--batch_size", "1", "--images_per_prompt", "1",
        ]
        subprocess.run(cmd, check=True)

    # Part B: Test why Pond Slider & Watersnake overfit to Box Turtle / Gopher Snake:
    # Compare Step 500 (40 ep) vs Step 750 (60 ep, mul 0.8) vs Step 1250 (100 ep, mul 1.0)
    # with explicit species color/pattern descriptions!
    overfit_in = TEST_DIR / "overfit_inputs"
    overfit_in.mkdir(exist_ok=True)
    overfit_species = [
        ("08_pond_slider", "pond_slider", "a red-eared pond slider turtle, dark olive-green aquatic carapace without yellow spots, bright red ear stripe behind eye, yellow and green striped swimming legs"),
        ("09_common_watersnake", "common_watersnake", "a common watersnake, dark brown and rusty-reddish crossbands on wet-looking dark scales, thick-bodied aquatic snake"),
        ("04_mule_deer", "mule_deer", "a mule deer buck, warm tan-brown coat, white rump patch, black-tipped tail, large mule ears, forked antlers, standing with all four hooves visible"),
    ]
    of_prompts = []
    for i, (inp_stem, slug, desc) in enumerate(overfit_species):
        import shutil
        shutil.copyfile(WORK / "inputs" / f"{inp_stem}.png", overfit_in / f"{i:02d}_{slug}.png")
        of_prompts.append(f"{desc}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42")
    of_pfile = TEST_DIR / "overfit_prompts.txt"
    of_pfile.write_text("\n".join(of_prompts), encoding="utf-8")

    for tag, ckpt, mul, st in [
        ("step500_m085_s064", "animl_style_v4-step00000500.safetensors", "0.85", "0.64"),
        ("step750_m080_s066", "animl_style_v4-step00000750.safetensors", "0.80", "0.66"),
    ]:
        odir = TEST_DIR / f"of_{tag}"
        odir.mkdir(exist_ok=True)
        if len(list(odir.glob("*.png"))) < 3:
            subprocess.run([
                str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
                "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
                "--outdir", str(odir),
                "--network_module", "networks.lora",
                "--network_weights", str(LORA / "output" / ckpt),
                "--network_mul", mul,
                "--from_file", str(of_pfile),
                "--image_path", str(overfit_in),
                "--use_original_file_name",
                "--strength", st,
                "--W", "1024", "--H", "1024",
                "--steps", "28", "--sampler", "euler_a",
                "--scale", "7.0", "--bf16", "--sdpa",
                "--batch_size", "1", "--images_per_prompt", "1",
            ], check=True)

    # Clean all outputs and assemble comparison sheets
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    # Sheet 1: Duck Foot Completion (Flat vs Paddling vs Tucked)
    duck_clean = TEST_DIR / "duck_clean"
    duck_clean.mkdir(exist_ok=True)
    for p in duck_out.glob("*.png"):
        clean_and_center(p, duck_clean / p.name)

    cols = ["Original Flat Floating", "Paddling Seed Input", "Paddling Feet Output", "Tucked Seed Input", "Tucked Feet Output"]
    cw, ch = 340, 340
    W = len(cols) * cw
    H = 48 + 2 * (ch + 32)
    s_duck = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(s_duck)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 14, 12), t, fill=(20, 20, 20), font=font)

    for ri, (slug, common, _) in enumerate(ducks):
        orig_idx = 1 if slug == "northern_pintail" else 2
        y_top = 48 + ri * (ch + 32)
        d.rectangle([0, y_top, W, y_top + 32], fill=(232, 232, 236))
        d.text((14, y_top + 5), f"{common} ({slug}) — Floating vs Paddling vs Tucked Webbed Feet", fill=(10, 10, 10), font=font_sm)
        files = [
            WORK / "clean_str_0.76" / f"{orig_idx:02d}_{slug}.png",
            duck_in / f"{ri*2:02d}_{slug}_paddling.png",
            duck_clean / f"{ri*2:02d}_{slug}_paddling.png",
            duck_in / f"{ri*2+1:02d}_{slug}_tucked.png",
            duck_clean / f"{ri*2+1:02d}_{slug}_tucked.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            s_duck.paste(cell, (ci * cw, y_top + 32))
            d.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))

    s_duck.save(ARTIFACT_DIR / "duck_foot_completion_compare.jpg", quality=92)

    # Sheet 2: Overfit Diagnostic (Step 1250 mul 1.0 vs Step 750 mul 0.80 vs Step 500 mul 0.85)
    for tag in ["step500_m085_s064", "step750_m080_s066"]:
        cdir = TEST_DIR / f"clean_{tag}"
        cdir.mkdir(exist_ok=True)
        for p in (TEST_DIR / f"of_{tag}").glob("*.png"):
            clean_and_center(p, cdir / p.name)

    cols2 = ["Reference Photo", "Step 1250 (100 ep, Overfit)", "Step 750 (60 ep, mul 0.80)", "Step 500 (40 ep, mul 0.85)"]
    W2 = len(cols2) * cw
    H2 = 48 + 3 * (ch + 32)
    s_of = Image.new("RGB", (W2, H2), (248, 248, 248))
    d2 = ImageDraw.Draw(s_of)
    for ci, t in enumerate(cols2):
        d2.text((ci * cw + 14, 12), t, fill=(20, 20, 20), font=font)

    for ri, (inp_stem, slug, _) in enumerate(overfit_species):
        y_top = 48 + ri * (ch + 32)
        d2.rectangle([0, y_top, W2, y_top + 32], fill=(232, 232, 236))
        d2.text((14, y_top + 5), f"{slug} — Reducing Species Overfit (Earlier Epoch + Lower LoRA Multiplier + Species Markings)", fill=(10, 10, 10), font=font_sm)
        files = [
            WORK / f"{slug}_photo.png",
            WORK / "clean_str_0.76" / f"{inp_stem}.png",
            TEST_DIR / "clean_step750_m080_s066" / f"{ri:02d}_{slug}.png",
            TEST_DIR / "clean_step500_m085_s064" / f"{ri:02d}_{slug}.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            s_of.paste(cell, (ci * cw, y_top + 32))
            d2.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))

    s_of.save(ARTIFACT_DIR / "overfit_diagnostic_compare.jpg", quality=92)
    print("Done diagnostic sheets!", flush=True)


if __name__ == "__main__":
    main()
