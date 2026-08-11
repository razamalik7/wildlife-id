"""Batch photo-reference generation for a set of vetted species picks.

For each species in the picks file:
  1. download the chosen reference photo
  2. rembg cutout -> subject scaled to 76% fill on white 1024
  3. img2img with the style LoRA at each denoise strength
  4. build a side-by-side review sheet (photo | cutout | outputs)

Usage:
    python art-pipeline/generate_batch.py --picks art-pipeline/vetting/picks_overnight.json
                                          [--strengths 0.6 0.7 0.8] [--seed 42]
"""

import argparse
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from normalize import subject_bbox  # noqa: E402
# rembg is imported lazily in prep(): its numba dependency can be blocked by
# Windows App Control, and most runs only need already-built cutouts.

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
NEG_BASE = ("photo, photorealistic, pattern, multiple animals, tiling, text, "
            "watermark, border, full scene, landscape, sky, speckles, confetti, "
            "dots, paint splatter, glow, cast shadow, sticker, heavy black outline, "
            "glossy, shiny, oversaturated, 3d render, airbrushed, cropped body")

# Locked by the overnight grid search: v2 LoRA + matte language beat v1 +
# neutral wording decisively (wigeon identity fixed, muskrat gloss gone).
STYLE_CLAUSE = ("muted matte colors, soft painterly brushwork, flat gouache shading")

# Environmental vignette per the style bible: the house style gives swimmers
# a stylized water pool, waders legs-in-water, perchers/climbers a branch.
# Prompt clause + which default negatives to drop for that context.
CONTEXTS = {
    "terrestrial": ("neutral standing pose, plain white background",
                    "scenery, foliage, branch, water"),
    "floating":    ("resting on a small stylized pool of blue water with simple "
                    "ripple lines, white background around the water",
                    "scenery, foliage, branch"),
    "wading":      ("standing with legs in a shallow stylized pool of blue water "
                    "with ripple lines, white background around the water",
                    "scenery, foliage, branch"),
    "perching":    ("perched on a single bare branch segment, white background",
                    "scenery, water"),
    "climbing":    ("climbing a single branch segment, white background",
                    "scenery, water"),
    "rock":        ("perched on a small stylized rock, white background",
                    "scenery, foliage, branch, water"),
    "coiled":      ("coiled resting pose, plain white background",
                    "scenery, foliage, branch, water"),
}

SPECIES_CONTEXT = {
    "wood_duck": "floating", "northern_shoveler": "floating",
    "northern_pintail": "floating", "american_wigeon": "floating",
    "green-winged_teal": "floating",
    "american_robin": "terrestrial", "great_egret": "wading",
    "downy_woodpecker": "perching",
    "mule_deer": "terrestrial", "muskrat": "floating",
    "california_ground_squirrel": "rock", "american_pika": "rock",
    "western_fence_lizard": "rock", "pond_slider": "terrestrial",
    "gopher_snake": "coiled", "common_watersnake": "coiled",
    "gopher_tortoise": "terrestrial", "american_toad": "terrestrial",
    "spring_peeper": "perching", "eastern_red-backed_salamander": "terrestrial",
}


def prep(slug: str, url: str, work: Path) -> Path:
    photo_p = work / f"{slug}_photo.png"
    cut_p = work / f"{slug}_cutout.png"
    if not cut_p.exists():
        from u2net_matte import remove
        r = requests.get(url, timeout=60, headers={"User-Agent": "animlvision"})
        photo = Image.open(BytesIO(r.content)).convert("RGBA")
        photo.convert("RGB").save(photo_p)
        cut = remove(photo)
        box = subject_bbox(cut)
        subject = cut.crop(box)
        scale = int(1024 * 0.76) / max(subject.size)
        nw, nh = (max(1, round(d * scale)) for d in subject.size)
        subject = subject.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canvas.paste(subject, ((1024 - nw) // 2, (1024 - nh) // 2), subject)
        canvas.convert("RGB").save(cut_p)
    return cut_p


def generate(slug: str, common: str, cut_p: Path, strength: float, seed: int, outdir: Path, args_lora: str = "animl_style_v2") -> None:
    if list(outdir.glob("*.png")):
        return
    ctx_clause, extra_neg = CONTEXTS[SPECIES_CONTEXT.get(slug, "terrestrial")]
    prompt = (f"a {common.lower()}, full body wildlife illustration, single animal, "
              f"complete body with all limbs visible, three-quarter view, "
              f"{STYLE_CLAUSE}, {ctx_clause} --n {NEG_BASE}, {extra_neg}")
    subprocess.run([
        str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
        "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        "--outdir", str(outdir),
        "--network_module", "networks.lora",
        "--network_weights", str(LORA / "output" / f"{args_lora}.safetensors"),
        "--network_mul", "1.0",
        "--prompt", prompt,
        "--image_path", str(cut_p),
        "--strength", str(strength),
        "--W", "1024", "--H", "1024", "--steps", "32", "--sampler", "euler_a",
        "--scale", "7", "--seed", str(seed), "--bf16", "--sdpa",
        "--batch_size", "1", "--images_per_prompt", "2",
    ], check=True, capture_output=True)


def sheet(slug: str, work: Path, strengths: list[float], out: Path) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    panels = [("reference photo", work / f"{slug}_photo.png"),
              ("cutout input", work / f"{slug}_cutout.png")]
    for s in strengths:
        files = sorted((work / f"{slug}_str{s}").glob("*.png"))
        if files:
            panels.append((f"output @ {s}", files[0]))
    T = 480
    im_out = Image.new("RGB", (T * len(panels), T + 44), "white")
    d = ImageDraw.Draw(im_out)
    for i, (label, p) in enumerate(panels):
        im = Image.open(p)
        im.thumbnail((T, T), Image.LANCZOS)
        cell = Image.new("RGB", (T, T), "white")
        cell.paste(im, ((T - im.width) // 2, (T - im.height) // 2))
        im_out.paste(cell, (i * T, 44))
        d.text((i * T + 10, 8), f"{label}", fill="black", font=font)
    d.text((10, im_out.height - 0), "", fill="black")
    im_out.save(out / f"{slug}_sheet.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default="art-pipeline/vetting/picks_overnight.json")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.6, 0.7, 0.8])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", default="art-pipeline/batch_run")
    ap.add_argument("--lora", default="animl_style_v2")
    args = ap.parse_args()

    picks = json.loads(Path(args.picks).read_text())
    work = Path(args.work)
    sheets = work / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    for slug, pick in picks.items():
        if pick.get("status") == "skipped":
            continue
        try:
            cut_p = prep(slug, pick["original_url"], work)
            for s in args.strengths:
                generate(slug, pick["common_name"], cut_p, s, args.seed,
                         work / f"{slug}_str{s}", args.lora)
            sheet(slug, work, args.strengths, sheets)
            print(f"done {slug}", flush=True)
        except Exception as e:
            print(f"FAILED {slug}: {e}", flush=True)


if __name__ == "__main__":
    main()
