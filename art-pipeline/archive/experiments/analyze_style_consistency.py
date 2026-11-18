import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ART = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")
DS = Path("art-pipeline/lora/dataset")
PRE_BIRDS = Path("art-pipeline/batch_run_true_birds_snake_v2/clean_s066")
PRE_V6 = Path("art-pipeline/batch_run_v6_eval/clean_v6_step1250_s076")
PRE_SNAKE = Path("art-pipeline/batch_run_true_art_style_3_7_8/clean_v6_s052")
B2 = Path("art-pipeline/batch_run_fresh_20_batch2/clean_out")


def load_all_items() -> list[tuple[str, Path]]:
    items = []
    for sub in ["img_6x", "img_4x", "img_2x"]:
        for p in sorted((DS / sub).glob("*.png")):
            items.append((f"[v6 {sub}] {p.stem}", p))
    for fn in ["00_northern_shoveler.png", "01_northern_pintail.png", "02_american_wigeon.png", "03_green-winged_teal.png"]:
        items.append((f"[Pre-B1] {fn[3:-4]}", PRE_BIRDS / fn))
    for fn in [
        "04_mule_deer.png",
        "05_muskrat.png",
        "06_california_ground_squirrel.png",
        "07_american_pika.png",
        "08_pond_slider.png",
        "10_gopher_tortoise.png",
    ]:
        items.append((f"[Pre-B1] {fn[3:-4]}", PRE_V6 / fn))
    for fn, lbl in [
        ("00_opt3_13608867.png", "watersnake_opt3"),
        ("01_opt7_526919366.png", "watersnake_opt7"),
        ("02_opt8_13810420.png", "watersnake_opt8"),
    ]:
        items.append((f"[Pre-B1] {lbl}", PRE_SNAKE / fn))
    for p in sorted(ART.glob("fresh20_v2_*.png")):
        if "bat_ray_front" in p.name:
            continue
        short = p.stem.replace("fresh20_v2_", "")
        items.append((f"[B1-v2] {short}", p))
    for p in sorted(B2.glob("*.png")):
        items.append((f"[B2] {p.stem}", p))
    return items


def measure_style(p: Path) -> dict:
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        rgb = np.array(bg)
    else:
        rgb = np.array(im.convert("RGB"))
    fg = np.any(rgb < 245, axis=-1)

    fg_u8 = fg.astype(np.uint8) * 255
    erode_1 = cv2.erode(fg_u8, np.ones((3, 3), np.uint8)) > 0
    erode_4 = cv2.erode(fg_u8, np.ones((9, 9), np.uint8)) > 0
    erode_12 = cv2.erode(fg_u8, np.ones((25, 25), np.uint8)) > 0

    outer_ring = erode_1 & (~erode_4)
    inner_ring = erode_4 & (~erode_12)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    outer_l = float(np.median(gray[outer_ring])) if outer_ring.any() else 128.0
    inner_l = float(np.median(gray[inner_ring])) if inner_ring.any() else 128.0
    dark_outline_pct = float((gray[outer_ring] < 68.0).mean() * 100.0) if outer_ring.any() else 0.0
    outline_drop = inner_l - outer_l

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    internal_grad = float(grad[erode_4].mean()) if erode_4.any() else 0.0
    internal_ink_pct = float(((grad > 45.0) & (gray < 75.0) & erode_4).mean() / max(1e-5, erode_4.mean()) * 100.0)

    bil = cv2.bilateralFilter(rgb, d=9, sigmaColor=35, sigmaSpace=35)
    micro_diff = (
        float(np.abs(rgb.astype(np.float32) - bil.astype(np.float32)).mean(axis=-1)[erode_4].mean())
        if erode_4.any()
        else 0.0
    )

    return {
        "outer_l": outer_l,
        "inner_l": inner_l,
        "outline_drop": outline_drop,
        "dark_outline_pct": dark_outline_pct,
        "internal_grad": internal_grad,
        "internal_ink_pct": internal_ink_pct,
        "micro_diff": micro_diff,
        "rgb": rgb,
        "fg": fg,
    }


def main() -> None:
    items = load_all_items()
    stats = {}
    for idx in range(1, len(items) + 1):
        lbl, p = items[idx - 1]
        stats[idx] = (lbl, measure_style(p))

    og_bold = [3, 4, 6, 8, 9, 11, 17, 18, 19, 21, 22, 23, 24, 25]
    og_soft = [1, 2, 5, 7, 12, 14, 15, 16, 20]  # (excluding 10 & 13 since user excluded 10 & 13)
    new_pre_b1 = [26, 28, 29, 30, 31, 34, 36]
    new_b1_b2 = [41, 42, 43, 47, 51, 52, 53, 55, 56, 57, 62, 66, 68, 69, 72]

    for group_name, idxs in [
        ("1. OG Subset A: Bold Dark Ink Outline (Robin/Woodpecker/Newt/Owl/Peeper/BarkingFrog/Tortoise)", og_bold),
        ("2. OG Subset B: Soft Umber Line / Painterly (Marten/Macaque/Goat/Deer/Seal)", og_soft),
        ("3. Picked Pre-B1 SDXL (26, 28-31, 34, 36)", new_pre_b1),
        ("4. Picked B1-v2 & B2 SDXL (41-43, 47, 51-53, 55-57, 62, 66, 68-69, 72)", new_b1_b2),
    ]:
        print(f"=== {group_name} (n={len(idxs)}) ===")
        for k in ["outer_l", "outline_drop", "dark_outline_pct", "internal_grad", "internal_ink_pct", "micro_diff"]:
            vals = [stats[i][1][k] for i in idxs]
            print(f"  {k:18s}: mean={np.mean(vals):6.1f}  median={np.median(vals):6.1f}  min={np.min(vals):5.1f}  max={np.max(vals):5.1f}")
        for i in idxs:
            s = stats[i][1]
            print(
                f"    #{i:2d} {stats[i][0]:34s} | outerL={s['outer_l']:5.1f} drop={s['outline_drop']:+5.1f} "
                f"darkOut%={s['dark_outline_pct']:4.1f}% intGrad={s['internal_grad']:4.1f} intInk%={s['internal_ink_pct']:4.1f}% micro={s['micro_diff']:3.1f}"
            )

    # Build a 4-row visual comparison sheet with 2.5x edge-zoom insets so the user can see the exact outline & detail differences!
    groups_vis = [
        ("Row 1: OG Subset A — Bold Dark Ink Outlines (NBP/Gemini Anchors)", [3, 6, 8, 9, 11, 17, 18]),
        ("Row 2: OG Subset B — Thin/Soft Umber Outlines (Original Masters)", [1, 2, 5, 7, 12, 14, 15]),
        ("Row 3: Picked Pre-B1 SDXL Set (26, 28–31, 34, 36)", [26, 28, 29, 30, 31, 34, 36]),
        ("Row 4: Picked B1-v2 & B2 SDXL Set (Sample of 7 from 41–72)", [41, 47, 52, 55, 57, 66, 68]),
    ]

    cols = 7
    cell_w, cell_h, hdr_h = 240, 290, 42
    w = cols * cell_w
    h = len(groups_vis) * (cell_h + hdr_h)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 18)
        font_lbl = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    for g_i, (g_title, g_idxs) in enumerate(groups_vis):
        y_top = g_i * (cell_h + hdr_h)
        draw.rectangle([0, y_top, w, y_top + hdr_h], fill=(38, 42, 50))
        draw.text((14, y_top + 10), g_title, fill=(255, 255, 255), font=font_hdr)
        y0 = y_top + hdr_h
        for c_i, idx in enumerate(g_idxs):
            x0 = c_i * cell_w
            lbl, s = stats[idx]
            rgb = s["rgb"]
            fg = s["fg"]
            sheet.paste(Image.fromarray(rgb).resize((224, 175), Image.Resampling.LANCZOS), (x0 + 8, y0 + 26))

            # Pick an actual outer_ring contour pixel near the upper-middle/head of the animal so the zoom crop never hits empty white space!
            fg_u8 = fg.astype(np.uint8) * 255
            er1 = cv2.erode(fg_u8, np.ones((3, 3), np.uint8)) > 0
            er4 = cv2.erode(fg_u8, np.ones((9, 9), np.uint8)) > 0
            ring_yx = np.argwhere(er1 & (~er4))
            # Pick the contour point closest to (y = 35% from top of bbox, x = 65% from left of bbox)
            ys, xs = np.where(fg)
            target_y = ys.min() + 0.35 * (ys.max() - ys.min())
            target_x = xs.min() + 0.65 * (xs.max() - xs.min())
            dists = (ring_yx[:, 0] - target_y) ** 2 + (ring_yx[:, 1] - target_x) ** 2
            best_pt = ring_yx[int(np.argmin(dists))]
            cy, cx = int(best_pt[0]), int(best_pt[1])
            y1, y2 = max(0, cy - 32), min(1024, cy + 32)
            x1, x2 = max(0, cx - 72), min(1024, cx + 72)
            crop = Image.fromarray(rgb[y1:y2, x1:x2]).resize((224, 72), Image.Resampling.NEAREST)
            sheet.paste(crop, (x0 + 8, y0 + 208))
            draw.rectangle([x0 + 8, y0 + 208, x0 + 232, y0 + 280], outline=(120, 120, 120), width=1)
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(215, 215, 215), width=1)
            short_lbl = lbl.split("] ")[-1][:18]
            draw.text(
                (x0 + 6, y0 + 6),
                f"#{idx} {short_lbl} (out%={s['dark_outline_pct']:.0f}%)",
                fill=(20, 20, 20),
                font=font_lbl,
            )

    out_sheet = ART / "style_outline_detail_analysis.jpg"
    sheet.save(out_sheet, quality=92)
    print(f"Saved visual analysis sheet to {out_sheet}")


if __name__ == "__main__":
    main()
