"""Test Lever 1: Pre-flattening the reference cutout into matte gouache color zones
before running SDXL img2img, plus extracting visual traits from Wikipedia.
"""

import re
import subprocess
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
LEVER1_DIR = WORK / "lever1_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")


def pre_flatten_gouache(src_p: Path, dst_p: Path, k_colors: int = 12) -> None:
    """Convert a raw photo cutout on white into a matte, flat-color gouache base:
    1. Isolate non-white subject pixels.
    2. Compress harsh photographic highlights & deep black shadows toward a warm matte midtone range.
    3. Smooth micro-grit using Median + Box/Gaussian edge-preserving blend so fur/feather camera noise becomes flat brush zones.
    4. Quantize the subject colors into k distinct painted color steps.
    """
    im = Image.open(src_p).convert("RGB")
    arr = np.array(im).astype(np.float32)
    fg_mask = (arr < 248).any(axis=2)
    if not fg_mask.any():
        im.save(dst_p)
        return

    # Step 1: Compress harsh photo lighting (lift deep shadows, soften specular highlights)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    fg_lum = lum[fg_mask]
    p5, p95 = np.percentile(fg_lum, 5), np.percentile(fg_lum, 95)
    if p95 > p5 + 10:
        target_lum = 44.0 + (np.clip((fg_lum - p5) / (p95 - p5), 0.0, 1.0) ** 0.90) * (222.0 - 44.0)
        ratio = target_lum / np.maximum(fg_lum, 1.0)
        # Blend 65% compressed luminance so colors stay rich and matte without blowing out
        scale = 0.35 + 0.65 * ratio
        for c in range(3):
            arr[:, :, c][fg_mask] = np.clip(arr[:, :, c][fg_mask] * scale, 24, 242)

    matte_im = Image.fromarray(arr.astype(np.uint8))

    # Step 2: Iterative median filter (size 5 then 3) to flatten photographic fur/scale noise into gouache shapes
    smooth_im = matte_im.filter(ImageFilter.MedianFilter(size=5)).filter(ImageFilter.MedianFilter(size=5))

    # Step 3: Quantize the foreground colors using PIL's fast octree/mediancut quantizer on cropped subject
    # Put subject on a neutral mid-tone background before quantizing so white doesn't skew palette
    s_arr = np.array(smooth_im)
    s_arr[~fg_mask] = [255, 255, 255]
    q_im = Image.fromarray(s_arr).quantize(colors=k_colors + 1, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
    q_arr = np.array(q_im)

    # Blend 75% quantized steps + 25% smooth matte so color transitions are crisp painted steps, not posterized artifacts
    final_arr = (0.75 * q_arr.astype(np.float32) + 0.25 * s_arr.astype(np.float32)).astype(np.uint8)
    final_im = Image.fromarray(final_arr).filter(ImageFilter.MedianFilter(size=3))
    final_np = np.array(final_im)
    final_np[~fg_mask] = [255, 255, 255]
    Image.fromarray(final_np).save(dst_p)


def fetch_wiki_visual_summary(species_title: str) -> str:
    """Quick test fetching Wikipedia text for a species and extracting visual sentences."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": species_title,
        "redirects": 1,
        "format": "json",
    }
    r = requests.get(url, params=params, headers={"User-Agent": "animlvision/1.0"}, timeout=15)
    pages = r.json().get("query", {}).get("pages", {})
    for p in pages.values():
        ext = p.get("extract", "")
        if ext:
            return ext[:1500]
    return ""


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
    LEVER1_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = LEVER1_DIR / "inputs"
    in_dir.mkdir(exist_ok=True)

    # Test Wikipedia extract on 2 species just to verify fast local lookup
    for sp in ["Northern pintail", "Pond slider"]:
        txt = fetch_wiki_visual_summary(sp)
        print(f"Wiki [{sp}]: {len(txt)} chars fetched -> {txt[:180].replace(chr(10), ' ')}...")

    # Test Lever 1 on 5 diverse unseen species:
    # Mule Deer, California Ground Squirrel, American Pika, Pond Slider, Green-winged Teal
    test_cases = [
        ("04_mule_deer", "mule_deer", "Mule Deer",
         "animl style illustration, a mule deer buck, warm tan-brown coat, white throat and rump patch, black-tipped tail, large mule ears, forked antlers, standing full body with all four hooves visible, three-quarter view"),
        ("06_california_ground_squirrel", "california_ground_squirrel", "California Ground Squirrel",
         "animl style illustration, a california ground squirrel, mottled brown and grey fur with pale shoulder cape, buffy belly, bushy tail, standing upright with front paws tucked, three-quarter view"),
        ("07_american_pika", "american_pika", "American Pika",
         "animl style illustration, an american pika, small round-bodied mammal with soft greyish-brown and buff fur, rounded ears, no visible tail, crouching full body with four paws visible, three-quarter view"),
        ("08_pond_slider", "pond_slider", "Pond Slider",
         "animl style illustration, a red-eared pond slider turtle, oval olive-green aquatic carapace with pale scute seams, bright red stripe behind eye, yellow-striped green legs, swimming pose, three-quarter view"),
        ("03_green-winged_teal", "green-winged_teal", "Green-winged Teal",
         "animl style illustration, a male green-winged teal duck, chestnut head with iridescent green eye-patch, speckled pinkish-buff breast, vermiculated grey flanks, green wing speculum, standing full body with webbed feet, three-quarter view"),
    ]

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

    prompts = []
    for i, (inp_stem, slug, common, ptxt) in enumerate(test_cases):
        src_p = WORK / "inputs" / f"{inp_stem}.png"
        dst_p = in_dir / f"{i:02d}_{slug}.png"
        pre_flatten_gouache(src_p, dst_p, k_colors=12)
        prompts.append(f"{ptxt}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42")

    pfile = LEVER1_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    # Run at two settings:
    # 1) Step 750 (60 ep, mul 0.85, strength 0.62) — lets the pre-flattened gouache color zones shine through without overfit
    # 2) Step 1000 (80 ep, mul 0.85, strength 0.66) — slightly stronger LoRA brushstroke/outline finish
    runs = [
        ("s750_str062", "animl_style_v4-step00000750.safetensors", "0.85", "0.62"),
        ("s1000_str066", "animl_style_v4-step00001000.safetensors", "0.85", "0.66"),
    ]
    for tag, ckpt, mul, st in runs:
        odir = LEVER1_DIR / f"out_{tag}"
        odir.mkdir(exist_ok=True)
        if len(list(odir.glob("*.png"))) < len(test_cases):
            subprocess.run([
                str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
                "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
                "--outdir", str(odir),
                "--network_module", "networks.lora",
                "--network_weights", str(LORA / "output" / ckpt),
                "--network_mul", mul,
                "--from_file", str(pfile),
                "--image_path", str(in_dir),
                "--use_original_file_name",
                "--strength", st,
                "--W", "1024", "--H", "1024",
                "--steps", "28", "--sampler", "euler_a",
                "--scale", "7.0", "--bf16", "--sdpa",
                "--batch_size", "1", "--images_per_prompt", "1",
            ], check=True)
        cdir = LEVER1_DIR / f"clean_{tag}"
        cdir.mkdir(exist_ok=True)
        for p in odir.glob("*.png"):
            clean_and_center(p, cdir / p.name)

    # Assemble comparison sheet:
    # Raw Cutout | Pre-Flattened Gouache Input | Raw Cutout -> v4 (Old) | Pre-Flattened -> v4 (s750 @ 0.62) | Pre-Flattened -> v4 (s1000 @ 0.66)
    try:
        font = ImageFont.truetype("arialbd.ttf", 21)
        font_sm = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    cols = [
        "1. Raw Photo Cutout",
        "2. Pre-Flattened Input",
        "3. Old (Raw -> v4 @ 0.76)",
        "4. Flat -> s750 @ 0.62",
        "5. Flat -> s1000 @ 0.66",
    ]
    cw, ch = 340, 340
    W = len(cols) * cw
    H = 48 + len(test_cases) * (ch + 32)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 12, 12), t, fill=(20, 20, 20), font=font)

    for ri, (inp_stem, slug, common, _) in enumerate(test_cases):
        y_top = 48 + ri * (ch + 32)
        d.rectangle([0, y_top, W, y_top + 32], fill=(232, 232, 236))
        d.text((14, y_top + 5), f"{common} ({slug}) — Lever 1: Pre-Flattened Gouache Cutout + Detailed Species Prompt", fill=(10, 10, 10), font=font_sm)
        files = [
            WORK / "inputs" / f"{inp_stem}.png",
            in_dir / f"{ri:02d}_{slug}.png",
            WORK / "clean_str_0.76" / f"{inp_stem}.png",
            LEVER1_DIR / "clean_s750_str062" / f"{ri:02d}_{slug}.png",
            LEVER1_DIR / "clean_s1000_str066" / f"{ri:02d}_{slug}.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 32))
            d.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "lever1_preflatten_compare.jpg", quality=92)
    print("Saved lever1_preflatten_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
