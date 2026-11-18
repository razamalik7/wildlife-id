"""Remove the leftover cast shadows from Option 7 (526919366) and Option 8 (13810420):
1. On Option 7 (526919366):
   - Remove the dark log cast shadow below the coil/head on the bottom-right (`x=630..900, y=635..770`).
   - Also provide a version where the dark shadow-coil under the jaw in the s=0.45 output is cleanly trimmed right along the jaw/neck contour.
2. On Option 8 (13810420):
   - Remove the dark gravel cast-shadow bulge on the rightmost curve (`x=845..900, y=440..575`) and the dark shadow sliver under the upper-right bend (`x=665..795, y=462..500`).
3. Re-run pure SDXL (`s=0.45` and `s=0.50`) on the shadow-cleaned input cutouts AND provide the directly trimmed `s=0.45` outputs.
"""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
ALL8_DIR = Path(__file__).parent / "batch_run_all8_snakes"
POSE_DIR = Path(__file__).parent / "batch_run_all8_snakes_pose_faithful"
ANAT_DIR = Path(__file__).parent / "batch_run_all8_snakes_anatomy_neg"
FIX_DIR = Path(__file__).parent / "batch_run_opt7_opt8_shadow_free"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")


def erase_polygon_to_white(im: Image.Image, polygons: list[list[tuple[int, int]]], feather: float = 1.5) -> Image.Image:
    """Erase specified polygon regions to clean solid white (255, 255, 255) with a 1.5px anti-aliased edge."""
    rgb = im.convert("RGB")
    mask = Image.new("L", rgb.size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        draw.polygon(poly, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    white = Image.new("RGB", rgb.size, (255, 255, 255))
    return Image.composite(white, rgb, mask)


def recenter_on_1024(im: Image.Image) -> Image.Image:
    """Re-center non-white subject on 1024x1024 white canvas at 78% scale."""
    arr = np.array(im.convert("RGB"))
    # Subject is where pixel is not near-white
    non_white = np.any(arr < 245, axis=-1)
    ys, xs = np.where(non_white)
    if len(xs) == 0:
        return im
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    subj = im.crop(box)
    scale = int(1024 * 0.78) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (1024, 1024), (255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2))
    return canvas


def main() -> None:
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    in_clean_dir = FIX_DIR / "inputs"
    in_clean_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean Option 7 (06_opt7_526919366) input cutout:
    # Polygon A: remove the pitch-black cast shadow below the coil under the head (y > 655..660 on the right)
    opt7_shadow_below_coil = [
        (640, 634), (675, 648), (700, 656), (725, 660), (750, 662),
        (775, 662), (800, 660), (825, 655), (850, 646), (868, 634),
        (881, 616), (920, 616), (920, 800), (600, 800)
    ]
    # Polygon B (for output where head expanded to y=642 and coil below head became dark shadow):
    # Trim cleanly along the bottom of the jaw + right coil curve
    opt7_shadow_tight_jaw = [
        (595, 595), (625, 618), (650, 630), (675, 639), (700, 644),
        (725, 645), (750, 643), (775, 639), (800, 633), (825, 624),
        (850, 612), (870, 596), (883, 575), (920, 575), (920, 800), (580, 800)
    ]

    # 2. Clean Option 8 (07_opt8_13810420) input cutout & output:
    # Rightmost point shadow bulge (x=850..900, y=435..580)
    opt8_rightmost_shadow_input = [
        (850, 430), (858, 450), (862, 475), (861, 500), (855, 525),
        (845, 548), (828, 568), (805, 582), (920, 582), (920, 430)
    ]
    # Also clean the dark cast shadow on the underside of the upper-right bend in the input cutout
    opt8_under_bend_shadow_input = [
        (670, 455), (700, 466), (725, 472), (750, 474), (775, 475),
        (795, 480), (795, 502), (670, 502)
    ]
    # On the s=0.45 output of Option 8 (`07_opt8_13810420_pure_s045.png`), the rightmost shadow bulge is at x=862..900, y=445..535:
    opt8_rightmost_shadow_output = [
        (856, 435), (863, 455), (867, 480), (866, 505), (860, 530),
        (848, 555), (920, 555), (920, 435)
    ]

    # Apply direct shadow removal to the existing Column 3 (s=0.45) outputs first:
    opt7_s045 = Image.open(POSE_DIR / "clean_s045" / "06_opt7_526919366.png")
    opt7_s045_keep_coil = recenter_on_1024(erase_polygon_to_white(opt7_s045, [opt7_shadow_below_coil]))
    opt7_s045_tight_jaw = recenter_on_1024(erase_polygon_to_white(opt7_s045, [opt7_shadow_tight_jaw]))
    opt7_s045_keep_coil.save(ARTIFACT_DIR / "06_opt7_526919366_s045_no_shadow_a.png")
    opt7_s045_tight_jaw.save(ARTIFACT_DIR / "06_opt7_526919366_s045_no_shadow_b.png")

    opt8_s045 = Image.open(POSE_DIR / "clean_s045" / "07_opt8_13810420.png")
    opt8_s045_clean = recenter_on_1024(erase_polygon_to_white(opt8_s045, [opt8_rightmost_shadow_output]))
    opt8_s045_clean.save(ARTIFACT_DIR / "07_opt8_13810420_s045_no_shadow.png")

    # Also clean the input cutouts and run pure SDXL s=0.45 & s=0.50 on them!
    opt7_in = Image.open(ALL8_DIR / "inputs" / "06_opt7_526919366.png")
    opt7_in_clean = recenter_on_1024(erase_polygon_to_white(opt7_in, [opt7_shadow_below_coil]))
    opt7_in_clean.save(in_clean_dir / "06_opt7_526919366.png")

    opt8_in = Image.open(ALL8_DIR / "inputs" / "07_opt8_13810420.png")
    opt8_in_clean = recenter_on_1024(erase_polygon_to_white(opt8_in, [opt8_rightmost_shadow_input, opt8_under_bend_shadow_input]))
    opt8_in_clean.save(in_clean_dir / "07_opt8_13810420.png")

    # Run pure SDXL at s=0.45 on the shadow-cleaned inputs
    STYLE_CLAUSE = (
        "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
        "subtle low-contrast warm brown-umber ink contour lines, "
        "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
    )
    NEG = (
        "(two heads:1.5), (double headed:1.5), (multiple heads:1.5), (extra head:1.5), "
        "(weird bulges:1.4), (lumpy swollen body:1.4), (unrealistic shape:1.4), (non-smooth coils:1.4), "
        "(impossible geometry:1.4), (uneven pinched tapering:1.4), (claws, legs, limbs, toes:1.4), "
        "cast shadow, drop shadow, photo, photorealistic, glossy, specular shine, stark white stripes, white rings"
    )
    prompts = [
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, matte dark brown and warm tan banded body, brown head on the right with pale yellow-tan chin, {STYLE_CLAUSE} --n {NEG} --d 42",
        f"animl_gouache_style, a banded marsh serpent, full body wildlife illustration, single animal, matte slate-brown and tan body with subtle dark crossbands, raised head in center with pale cream throat, slender tapered tail on left, {STYLE_CLAUSE} --n {NEG} --d 42",
    ]
    pf = FIX_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"
    out_raw = FIX_DIR / "raw_s045"
    out_raw.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
        "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        "--outdir", str(out_raw),
        "--network_module", "networks.lora",
        "--network_weights", str(lora_1250),
        "--network_mul", "0.88",
        "--max_embeddings_multiples", "2",
        "--from_file", str(pf),
        "--image_path", str(in_clean_dir),
        "--use_original_file_name",
        "--strength", "0.45",
        "--W", "1024", "--H", "1024",
        "--steps", "28", "--sampler", "euler_a",
        "--scale", "5.0", "--bf16", "--sdpa",
        "--batch_size", "1", "--images_per_prompt", "1",
    ]
    subprocess.run(cmd, check=True)

    for png in sorted(out_raw.glob("*.png")):
        im = Image.open(png).convert("RGBA")
        cut = remove(im)
        arr = np.array(cut)
        arr[arr[:, :, 3] < 165, 3] = 0
        cut = Image.fromarray(arr)
        box = subject_bbox(cut)
        subj = cut.crop(box)
        scale = int(1024 * 0.78) / max(subj.size)
        nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
        subj = subj.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
        canvas.convert("RGB").save(ARTIFACT_DIR / f"{png.stem}_regenerated_s045.png")

    print("Done shadow removal and regeneration!", flush=True)


if __name__ == "__main__":
    main()
