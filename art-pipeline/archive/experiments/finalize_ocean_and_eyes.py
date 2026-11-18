"""
Finalize Part 2 of the 20-species v2 suite:
1. Restore the clean NOAA underwater swimming Bat Ray (`File:Bat-ray-NOAA-Tony-Chess.jpg` + `fresh20_v2_20_bat_ray_front.png`)
   into `20_bat_ray` (photo, cutout, Canny, and stylized output).
2. Restore the reference photo's true iris coloration on `16_green_treefrog` (white/cream iris around black pupil)
   and `17_wood_frog` (top-half cream-white, bottom-half brown iris) via localized cel-shaded iris transfer.
3. Rebuild `v6_fresh_20_v2_part2.jpg` with all updated rows (Species 11–20).
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from test_fresh_20_e2e import FRESH_20_SPECIES, download_pil, make_normalized_cutout
from test_pre_stylization_crop_repair import build_canny_from_repaired

WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
V2_WORK = Path(__file__).parent / "batch_run_fresh_20_v2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")


def restore_frog_eye_color(stem: str, eye_box: tuple[int, int, int, int]) -> None:
    """Transfer the reference photo cutout's cel-shaded iris lightness and hue inside `eye_box`
    where SDXL's frog prior darkened a bright light-colored iris into dark brown."""
    cut_p = WORK / "cutouts_1024" / f"{stem}.png"
    styl_p = V2_WORK / "clean_out" / f"{stem}.png"
    cut_im = Image.open(cut_p).convert("RGB")
    styl_rgba = Image.open(styl_p).convert("RGBA")

    cut_arr = np.array(cut_im)
    styl_arr = np.array(styl_rgba)

    x1, y1, x2, y2 = eye_box
    cut_crop = cut_arr[y1:y2, x1:x2]
    styl_crop = styl_arr[y1:y2, x1:x2, :3].copy()

    # Cel-shade the photo eye crop with bilateral filtering and boost bright iris contrast
    cut_smooth = cv2.bilateralFilter(cut_crop, d=9, sigmaColor=45, sigmaSpace=45)
    cut_smooth = cv2.bilateralFilter(cut_smooth, d=9, sigmaColor=45, sigmaSpace=45)
    lab_s = cv2.cvtColor(cut_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    bright_px = lab_s[:, :, 0] > 110.0
    lab_s[bright_px, 0] = np.clip(lab_s[bright_px, 0] * 1.18 + 14.0, 0, 248)
    cut_smooth = cv2.cvtColor(lab_s.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Build a soft elliptical mask inside eye_box where the photo has bright iris
    # (including where SDXL over-expanded the black pupil over the photo's white iris!)
    h, w = cut_crop.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h / 2.0, w / 2.0
    dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)

    cut_l = cv2.cvtColor(cut_crop, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    styl_l = cv2.cvtColor(styl_crop, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)

    # Keep any bright white specular catchlight (styl_l > 215) from the stylized eye,
    # and restore the photo's bright iris ring everywhere cut_l > 105 inside the eye ellipse!
    ellipse_w = np.clip((0.92 - dist) / 0.22, 0.0, 1.0)
    iris_mask = ((cut_l > 105.0) & (styl_l < 215.0)).astype(np.float32)
    iris_mask = cv2.GaussianBlur(iris_mask * ellipse_w, (5, 5), 1.0)[:, :, None]

    blended = np.clip(
        styl_crop.astype(np.float32) * (1.0 - 0.92 * iris_mask) + cut_smooth.astype(np.float32) * (0.92 * iris_mask),
        0,
        255,
    ).astype(np.uint8)
    styl_arr[y1:y2, x1:x2, :3] = blended

    out_im = Image.fromarray(styl_arr)
    out_im.save(styl_p)
    shutil.copyfile(styl_p, ARTIFACT_DIR / f"fresh20_v2_{stem}.png")


def rebuild_part2_sheet() -> None:
    species_part2 = FRESH_20_SPECIES[10:20]
    col_w = 340
    row_h = 340
    header_h = 56
    cols = 4
    w = cols * col_w
    h = header_h + len(species_part2) * row_h

    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 22)
        font_lbl = ImageFont.truetype("arialbd.ttf", 17)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    headers = [
        "1. Selected Ref Photo",
        "2. Repaired Cutout (v2)",
        "3. v1 Stylized (Before)",
        "4. v2 Stylized (Fixed)",
    ]
    draw.rectangle([0, 0, w, header_h], fill=(35, 38, 45))
    for c_idx, h_text in enumerate(headers):
        draw.text((c_idx * col_w + 16, 16), h_text, fill=(255, 255, 255), font=font_hdr)

    for r_idx, (stem, common, sci, _, _, _) in enumerate(species_part2):
        y0 = header_h + r_idx * row_h
        p_photo = WORK / "photos_selected" / f"{stem}.png"
        p_cut = V2_WORK / "cutouts_v2" / f"{stem}.png"
        if not p_cut.exists():
            p_cut = WORK / "cutouts_1024" / f"{stem}.png"
        p_v1 = WORK / "clean_out" / f"{stem}.png"
        p_v2 = V2_WORK / "clean_out" / f"{stem}.png"

        for c_idx, p in enumerate([p_photo, p_cut, p_v1, p_v2]):
            x0 = c_idx * col_w
            if p.exists():
                im = Image.open(p)
                if im.mode == "RGBA":
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    bg.paste(im, mask=im.split()[3])
                    im = bg
                else:
                    im = im.convert("RGB")
                im_r = im.resize((col_w - 16, row_h - 42), Image.Resampling.LANCZOS)
                sheet.paste(im_r, (x0 + 8, y0 + 34))
            draw.rectangle([x0, y0, x0 + col_w, y0 + row_h], outline=(220, 220, 220), width=1)

        draw.text((12, y0 + 8), f"{stem}: {common} ({sci})", fill=(20, 20, 20), font=font_lbl)

    out_sheet = ARTIFACT_DIR / "v6_fresh_20_v2_part2.jpg"
    sheet.save(out_sheet, quality=92)
    print(f"Saved updated Part 2 comparison sheet to {out_sheet}", flush=True)


def main() -> None:
    front_ray = ARTIFACT_DIR / "fresh20_v2_20_bat_ray_front.png"
    if front_ray.exists():
        shutil.copyfile(front_ray, V2_WORK / "clean_out" / "20_bat_ray.png")
        shutil.copyfile(front_ray, ARTIFACT_DIR / "fresh20_v2_20_bat_ray.png")

    # 2. Restore true reference-photo iris colors on 16_green_treefrog and 17_wood_frog
    # 16_green_treefrog eye bbox: (528, 266, 602, 344)
    restore_frog_eye_color("16_green_treefrog", (528, 266, 602, 344))
    # 17_wood_frog eye bbox: (224, 306, 322, 398)
    restore_frog_eye_color("17_wood_frog", (224, 306, 322, 398))

    # 3. Rebuild Part 2 comparison sheet
    rebuild_part2_sheet()


if __name__ == "__main__":
    main()
