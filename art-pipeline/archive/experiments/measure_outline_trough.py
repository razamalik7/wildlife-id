"""
Halo-Invariant 1D Rebound-Trough Outline Detector:
Why previous methods failed:
1. RGBA/anti-aliased images have a 2–4px near-white halo (`L = 230..253`) outside the true edge, shifting fixed `d=1..3px` bands into white space.
2. Light-colored animals (`#47 scrub_jay`, `#55 wood_frog`) often have a colored/umber outline (`L_trough = 100..145` against an `L_body = 190..235` interior — a +85 luminance rebound!), which a fixed `L < 68` threshold misses.
3. Conversely, a dark stripe or dark animal (`L_edge = 40, L_body = 40`) has zero rebound (`L_body - L_trough = 0`) and is NOT an outline.

How this detector works:
1. Starts the silhouette strictly at `L < 215` (the 50% contrast crossing), completely ignoring any outer white/grey halo (`215..255`), and fills internal holes so white feathers (`#9 woodpecker`, `#17 snowy_owl`) are inside the silhouette.
2. At each perimeter segment, walks inward along the normal from `d = 0.5 px` to `d = 12 px`:
   - Finds the local minimum `L_trough` in the outer edge zone `d in [0.5 .. 5.0 px]` at distance `d_min`.
   - Finds the immediate interior body level `L_body` (75th percentile of `L` in `d in [d_min + 2.0 .. d_min + 7.5 px]`, i.e., strictly *behind* the trough).
   - Computes the True Trough Rebound:
       Rebound = L_body - L_trough
     If the profile monotonically darkens into a dark body/stripe without rebounding (`L_body <= L_trough + 10`), `Rebound ~ 0` (NOT an outline!).
3. Classifies every non-black boundary point (`L_body > 75`) into 3 clear physical categories:
   - BLACK / DARK CHARCOAL OUTLINE (Green): `Rebound >= 18` AND `L_trough < 70`
   - COLORED / WARM UMBER OUTLINE (Orange): `Rebound >= 15` AND `70 <= L_trough < 150`
   - NO OUTLINE / LINELESS EDGE (Red): `Rebound < 15` (body color meets background with no darker bounding line)
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze_style_consistency import load_all_items, ART

PICKED_45 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    11, 12,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    28, 29, 30, 31,
    34, 36,
    41, 42, 43,
    47,
    51, 52, 53,
    55, 56, 57,
    62, 66, 68, 69, 76,
]


def get_halo_free_silhouette(p: Path) -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        rgb = np.array(bg)
    else:
        rgb = np.array(im.convert("RGB"))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Threshold at L < 218 so any 220..255 anti-aliased white halo is outside d=0!
    fg_raw = (gray < 218).astype(np.uint8)
    # Close small gaps up to 11px so white feathers on the perimeter are bridged into the solid silhouette
    closed = cv2.morphologyEx(fg_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(fg_raw)
    for c in contours:
        if cv2.contourArea(c) > 400:
            cv2.drawContours(solid, [c], -1, 1, thickness=-1)
    return rgb, solid


def analyze_rebound_outline(p: Path) -> dict:
    rgb, solid = get_halo_free_silhouette(p)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    # Smooth distance field before taking gradient so normal vectors are smooth and never jitter at 1px notches
    dist_smooth = cv2.GaussianBlur(dist, (7, 7), 1.5)
    gx = cv2.Sobel(dist_smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist_smooth, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx = gx / mag
    ny = gy / mag

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c.reshape(-1, 2) for c in contours if len(c) >= 40])[::3]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    d_steps = np.linspace(0.5, 11.5, 23)  # every 0.5px

    rebounds = []
    trough_ls = []
    cats = []  # 1=Black/Charcoal outline, 2=Colored/Umber outline, 3=No outline
    vis_pts = []  # (x, y, cat) where 0=Dark body excluded

    for x0, y0 in pts:
        n_x = float(nx[y0, x0])
        n_y = float(ny[y0, x0])
        if abs(n_x) + abs(n_y) < 0.1:
            continue
        xs = np.clip(np.round(x0 + n_x * d_steps).astype(int), 0, w - 1)
        ys = np.clip(np.round(y0 + n_y * d_steps).astype(int), 0, h - 1)
        if dist[ys[-1], xs[-1]] < 3.5:
            continue

        prof = gray[ys, xs]
        # Find local trough minimum in outer edge zone d in [0.5 .. 4.5 px] (indices 0..8)
        idx_min = int(np.argmin(prof[:9]))
        l_trough = float(prof[idx_min])

        # Look strictly BEHIND the trough: indices [idx_min + 4 .. idx_min + 14] (2px to 7px deeper inside than the trough!)
        start_in = min(len(prof) - 3, idx_min + 4)
        end_in = min(len(prof), idx_min + 15)
        interior_slice = prof[start_in:end_in]
        l_body = float(np.percentile(interior_slice, 70))

        # If both the edge and the interior right behind it are dark (< 80), that's a dark animal/feather/stripe, NOT an outline!
        if l_body < 80.0 and l_trough < 75.0:
            vis_pts.append((x0, y0, 0))
            continue

        rebound = l_body - l_trough
        rebounds.append(rebound)
        trough_ls.append(l_trough)

        if rebound >= 18.0 and l_trough < 72.0:
            cat = 1  # Crisp Black / Dark Charcoal outline
        elif rebound >= 14.0 and l_trough < 152.0:
            cat = 2  # Colored / Warm Umber / Shaded outline
        else:
            cat = 3  # Lineless edge (no rebound trough)
        cats.append(cat)
        vis_pts.append((x0, y0, cat))

    cats_arr = np.array(cats) if cats else np.array([3])
    reb_arr = np.array(rebounds) if rebounds else np.array([0.0])
    tl_arr = np.array(trough_ls) if trough_ls else np.array([128.0])

    black_line_pct = float((cats_arr == 1).mean() * 100.0)
    umber_line_pct = float((cats_arr == 2).mean() * 100.0)
    total_line_pct = black_line_pct + umber_line_pct
    no_line_pct = float((cats_arr == 3).mean() * 100.0)
    med_rebound = float(np.median(reb_arr))
    outlined = (cats_arr == 1) | (cats_arr == 2)
    med_trough_l = float(np.median(tl_arr[outlined])) if outlined.any() else float(np.median(tl_arr))

    return {
        "black_pct": black_line_pct,
        "umber_pct": umber_line_pct,
        "total_pct": total_line_pct,
        "no_line_pct": no_line_pct,
        "med_rebound": med_rebound,
        "med_trough_l": med_trough_l,
        "vis_pts": vis_pts,
        "rgb": rgb,
    }


def main() -> None:
    items = load_all_items()
    results = {}
    print(
        f"{'ID':>3s} {'Species':34s} | {'BlackLine%':>10s} {'UmberLine%':>10s} {'TotalLine%':>10s} {'NoLine%':>8s} {'Rebound':>8s} {'Line_L':>7s}"
    )
    print("-" * 98)
    for idx in PICKED_45:
        lbl, p = items[idx - 1]
        r = analyze_rebound_outline(p)
        results[idx] = (lbl, r)
        print(
            f"#{idx:2d} {lbl:34s} | {r['black_pct']:9.1f}% {r['umber_pct']:9.1f}% {r['total_pct']:9.1f}% "
            f"{r['no_line_pct']:7.1f}% {r['med_rebound']:+7.1f} {r['med_trough_l']:7.1f}"
        )

    demo_ids = [9, 17, 11, 76, 1, 15, 6, 18, 30, 41, 47, 55, 57, 66, 68, 69]
    cols, rows = 4, 4
    cw, ch = 340, 340
    sheet = Image.new("RGB", (cols * cw, rows * ch + 60), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 17)
        font_lbl = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    draw.rectangle([0, 0, cols * cw, 56], fill=(35, 38, 45))
    draw.text(
        (14, 8),
        "Halo-Invariant Rebound-Trough Outline Detector (Requires V-Valley Rebound Inside Edge!)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    draw.text(
        (14, 32),
        "GREEN = Black/Charcoal Line (L<72)  |  ORANGE = Colored/Umber Line (L=72..152)  |  RED = No Line (No Rebound)  |  BLUE = Dark Body (Excluded)",
        fill=(210, 225, 245),
        font=font_lbl,
    )

    for i, idx in enumerate(demo_ids):
        lbl, r = results[idx]
        vis = r["rgb"].copy()
        for x0, y0, st in r["vis_pts"]:
            if st == 0:
                color = (70, 130, 230)  # Blue: Dark body interior (excluded)
            elif st == 1:
                color = (20, 195, 55)   # Green: Black/charcoal outline
            elif st == 2:
                color = (245, 155, 15)  # Orange: Colored/umber outline
            else:
                color = (235, 45, 45)   # Red: No outline
            cv2.circle(vis, (int(x0), int(y0)), 4, color, -1)

        row_i, col_i = divmod(i, cols)
        x_corner = col_i * cw
        y_corner = 56 + row_i * ch
        sheet.paste(Image.fromarray(vis).resize((cw - 16, ch - 52), Image.Resampling.LANCZOS), (x_corner + 8, y_corner + 42))
        draw.rectangle([x_corner, y_corner, x_corner + cw, y_corner + ch], outline=(215, 215, 215), width=1)
        short = lbl.split("] ")[-1][:22]
        draw.text((x_corner + 8, y_corner + 6), f"#{idx} {short}", fill=(20, 20, 20), font=font_lbl)
        draw.text(
            (x_corner + 8, y_corner + 23),
            f"BlkLine={r['black_pct']:.0f}%  UmbLine={r['umber_pct']:.0f}%  NoLine={r['no_line_pct']:.0f}%  Reb={r['med_rebound']:+.0f}",
            fill=(70, 70, 70),
            font=font_lbl,
        )

    out_p = ART / "inward_normal_outline_verification.jpg"
    sheet.save(out_p, quality=92)
    print(f"Saved verification sheet to {out_p}")

    # Also build the full 45-tile cohesion board (9x5) with #76 and the new rebound-trough metrics
    cols45, rows45 = 9, 5
    tw, th = 220, 235
    board45 = Image.new("RGB", (cols45 * tw, rows45 * th + 58), (255, 255, 255))
    d45 = ImageDraw.Draw(board45)
    d45.rectangle([0, 0, cols45 * tw, 54], fill=(30, 34, 42))
    d45.text(
        (14, 7),
        "Your 45 Picked Candidates (#76 Corrected) — Halo-Invariant Rebound-Trough Outline Metrics",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    d45.text(
        (14, 31),
        "Perimeter Overlay: GREEN = Black/Charcoal Line (L<72) | ORANGE = Colored/Umber Line (L=72..152) | RED = Lineless Edge | BLUE = Dark Body Excluded",
        fill=(210, 225, 245),
        font=font_lbl,
    )
    for k, idx in enumerate(PICKED_45):
        lbl, r = results[idx]
        vis = r["rgb"].copy()
        for x0, y0, st in r["vis_pts"]:
            if st == 0:
                c = (70, 130, 230)
            elif st == 1:
                c = (20, 195, 55)
            elif st == 2:
                c = (245, 155, 15)
            else:
                c = (235, 45, 45)
            cv2.circle(vis, (int(x0), int(y0)), 4, c, -1)
        ri, ci = divmod(k, cols45)
        xc = ci * tw
        yc = 54 + ri * th
        board45.paste(Image.fromarray(vis).resize((tw - 12, th - 48), Image.Resampling.LANCZOS), (xc + 6, yc + 42))
        # Border color indicates dominant outline type: Green (Black>=50%), Orange (Total>=80% via Umber), Red (NoLine>20%)
        if r["black_pct"] >= 50.0:
            bcol = (30, 175, 65)
        elif r["total_pct"] >= 80.0:
            bcol = (235, 145, 20)
        else:
            bcol = (225, 60, 60)
        d45.rectangle([xc + 1, yc + 1, xc + tw - 2, yc + th - 2], outline=bcol, width=2)
        short = lbl.split("] ")[-1][:18]
        d45.text((xc + 6, yc + 5), f"#{idx} {short}", fill=(20, 20, 20), font=font_lbl)
        d45.text(
            (xc + 6, yc + 22),
            f"Blk:{r['black_pct']:.0f}% Umb:{r['umber_pct']:.0f}% NoL:{r['no_line_pct']:.0f}%",
            fill=(65, 65, 65),
            font=font_lbl,
        )
    out45 = ART / "picked_45_cohesion_board.jpg"
    board45.save(out45, quality=92)
    print(f"Saved 45-candidate board to {out45}")


if __name__ == "__main__":
    main()
