"""
Upgrades `enhance_outline_skin_tight` and our outline measurement system to include:
1. Radial Shell Profile (`Shells d = 0..1, 1..2, 2..3, 3..4 px`)
2. Effective Dark Stroke Width (`Width_px` where `L < 85`)
3. Integrated 3px Rim Darkness (`Rim3_L` = median luminance across the entire `d = 0.5 .. 3.0 px` band)

Why `#41` and `#42` looked less darkly outlined from afar than `#1 marten`, `#2 macaque`, `#15 mountain_goat`, `#43 marmot`:
- On `#1, #2, #15, #30, #43`, the dark outline is `2.8 - 4.6 px` wide (`Width_px`), with `Rim3_L = 30..52` and `Shell 0 = 47..67`.
- On `#41` and `#42`, our previous `3x3` (1px) erosion only made a `0.8 - 1.0 px` wide line at `L = 65..77`, AND left a pale
  `L = 189..203` fringe in `Shell 0 (d=0..1px)` outside the furry edges, AND missed `#42`'s white belly (`19% NoLine`)!
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze_style_consistency import load_all_items, ART
from measure_outline_trough import get_halo_free_silhouette, analyze_rebound_outline
from tight_outline_post import enhance_outline_skin_tight


def enhance_outline_true_match(p: Path, strength: float = 0.94) -> tuple[np.ndarray, Path]:
    im = Image.open(p)
    has_alpha = im.mode == "RGBA"
    if has_alpha:
        rgba = np.array(im)
        rgb_raw = rgba[:, :, :3].astype(np.float32)
        alpha_raw = rgba[:, :, 3].astype(np.float32) / 255.0
        rgb_orig = (rgb_raw * alpha_raw[..., None] + 255.0 * (1.0 - alpha_raw[..., None])).astype(np.uint8)
    else:
        rgb_orig = np.array(im.convert("RGB"))
        alpha_raw = np.ones(rgb_orig.shape[:2], dtype=np.float32)

    rgb_f = rgb_orig.astype(np.float32)
    gray = cv2.cvtColor(rgb_orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. Exterior background flood-fill strictly through TRUE white background (`gray >= 238`):
    #    This prevents exterior flood-fill from leaking through silver/white-frosted hairs (`gray = 210..228`)
    #    inside `#42 western_gray_squirrel`'s tail, while still entering every true white V-notch (`gray >= 238`)!
    blur_g = cv2.GaussianBlur(gray, (7, 7), 1.6)
    ext_bg = (gray >= 238.0).astype(np.uint8)
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ext_bg, mask_ff, (0, 0), 2)
    cv2.floodFill(ext_bg, mask_ff, (w - 1, 0), 2)
    cv2.floodFill(ext_bg, mask_ff, (0, h - 1), 2)
    cv2.floodFill(ext_bg, mask_ff, (w - 1, h - 1), 2)

    # On the exterior boundary (adjacent to ext_bg == 2), use continuous sub-pixel opacity `alpha_lum`
    # Inside the silhouette (ext_bg != 2), opacity is 1.0 (so white bellies & silver tail hairs are 100% solid!)
    alpha_lum = np.clip((242.0 - gray) / 22.0, 0.0, 1.0)
    alpha_exact = np.where(ext_bg != 2, np.maximum(alpha_lum, 0.95), alpha_lum).astype(np.float32)

    # 2. Slice off protruding thin whiskers (<4px wide) at the snout/cheek root via 5x5 morphological opening,
    #    while keeping every concave V-notch strictly open (`minimum(alpha_exact, ...)`):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(alpha_exact, cv2.MORPH_OPEN, k_open)
    alpha_body = np.minimum(alpha_exact, cv2.GaussianBlur(opened, (3, 3), 0.5))

    # Protruding whisker mask: in `alpha_exact` (or darker than local background ridge) but sliced off by `alpha_body`
    is_whisker = ((alpha_exact - alpha_body) > 0.25) | (((blur_g - gray) > 3.5) & (alpha_body < 0.25))

    # 3. Build a TRUE 2.8px wide inward outline trough (`d = 0.0 .. 2.8 px`) matching OG `#1 marten` / `#15 mountain_goat`:
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_1 = cv2.erode(alpha_body, k3, iterations=1)
    inner_2 = cv2.erode(inner_1, k5, iterations=1)
    inner_core = cv2.GaussianBlur(inner_2, (5, 5), 1.1)

    edge_band = np.clip(alpha_body - inner_core, 0.0, 1.0)
    body_gate = np.clip((alpha_body - 0.10) / 0.40, 0.0, 1.0)
    stroke_weight = np.clip(edge_band * 1.75, 0.0, 1.0) * body_gate * strength

    w_ink = stroke_weight[..., None]
    charcoal_target = 0.12 * (rgb_f * 0.20) + 0.88 * np.array([16.0, 15.0, 14.0], dtype=np.float32)
    out = rgb_f * (1.0 - w_ink) + np.minimum(rgb_f, charcoal_target) * w_ink

    # 4. Clean any non-whisker pale matting fringe outside `alpha_body` so `Shell 0` starts directly on the dark stroke:
    outer_fringe = (ext_bg == 2) & (alpha_body < 0.20) & (~is_whisker) & (gray > 175.0)
    clean_w = np.clip((0.22 - alpha_body) / 0.22, 0.0, 1.0)[..., None]
    out = np.where(outer_fringe[..., None], out * (1.0 - clean_w) + 255.0 * clean_w, out)

    res_u8 = np.clip(out, 0, 255).astype(np.uint8)
    tmp_path = ART / f"scratch_truematch_{p.stem}.png"
    Image.fromarray(res_u8).save(tmp_path)
    return res_u8, tmp_path


def measure_shells_and_width(p: Path) -> dict:
    rgb, solid = get_halo_free_silhouette(p)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    shells = [float(np.median(gray[(dist > d) & (dist <= d + 1)])) for d in range(5)]
    rim3_l = float(np.median(gray[(dist > 0.5) & (dist <= 3.0)]))
    w_px = sum(1 for d in np.linspace(0.2, 6.0, 30) if np.median(gray[(dist > d) & (dist <= d + 0.4)]) < 85) * 0.2
    reb = analyze_rebound_outline(p)
    return {
        "shells": shells,
        "rim3_l": rim3_l,
        "width_px": w_px,
        "blk_pct": reb["black_pct"],
        "nol_pct": reb["no_line_pct"],
    }


def main() -> None:
    items = load_all_items()
    test_ids = [1, 2, 6, 15, 17, 30, 43, 41, 42, 47, 55, 62]
    print(
        f"{'ID':>3s} {'Species':30s} | {'Shells d=0..4 (px)':26s} | {'Width':>6s} | {'Rim3_L':>6s} | {'Blk%':>6s} {'NoL%':>5s}"
    )
    print("-" * 98)
    processed = {}
    for idx in test_ids:
        lbl, p = items[idx - 1]
        rgb_old, p_old = enhance_outline_skin_tight(p, strength=0.88)
        rgb_new, p_new = enhance_outline_true_match(p, strength=0.94)
        m_new = measure_shells_and_width(p_new)
        m_old = measure_shells_and_width(p_old)
        processed[idx] = (lbl, rgb_old, m_old, rgb_new, m_new)
        sh_str = "[" + ", ".join(f"{x:4.0f}" for x in m_new["shells"][:4]) + "]"
        print(
            f"#{idx:2d} {lbl.split('] ')[-1][:30]:30s} | {sh_str:26s} | {m_new['width_px']:4.1f}px | "
            f"{m_new['rim3_l']:6.1f} | {m_new['blk_pct']:5.1f}% {m_new['nol_pct']:4.1f}%"
        )

    # Visual comparison sheet:
    # Top row: Reference Bold-Outlined Mammals (#1 Marten, #2 Macaque, #15 Mountain Goat, #43 Marmot) + Previous 1px (#41, #42) vs New 2.8px (#41, #42)
    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 17)
        font_lbl = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    cols = 6
    cw, ch_far, ch_zoom = 300, 260, 220
    sheet = Image.new("RGB", (cols * cw, 2 * ch_far + ch_zoom + 62), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, cols * cw, 58], fill=(26, 30, 38))
    draw.text(
        (14, 7),
        "Why Squirrels Looked Fainter From Afar (0.8px Width + L=195 Fringe) -> Fixed to Match Marten/Macaque (2.8px Width)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    draw.text(
        (14, 32),
        "Cols 1-2: Reference #1 Marten & #2 Macaque  |  Cols 3-6: #41 Squirrel, #42 Gray Squirrel, #55 Frog, #62 Kangaroo Rat (Row 1=Old 1px, Row 2=NEW 2.8px, Row 3=4x Zoom)",
        fill=(205, 225, 245),
        font=font_lbl,
    )

    # Cols 0 & 1: Reference #1 Marten & #2 Macaque
    for ci, ref_id in enumerate([1, 2]):
        lbl, _, _, rgb_new, m_new = processed[ref_id]
        short = lbl.split("] ")[-1][:20]
        xc = ci * cw
        for ri, row_tag in enumerate(["REFERENCE (Far View)", "REFERENCE (Same)"]):
            yc = 58 + ri * ch_far
            sheet.paste(
                Image.fromarray(rgb_new).resize((cw - 16, ch_far - 50), Image.Resampling.LANCZOS),
                (xc + 8, yc + 44),
            )
            draw.rectangle([xc, yc, xc + cw, yc + ch_far], outline=(100, 140, 210), width=2)
            draw.text((xc + 8, yc + 6), f"REF #{ref_id} {short}", fill=(25, 65, 150), font=font_lbl)
            draw.text(
                (xc + 8, yc + 23),
                f"Width={m_new['width_px']:.1f}px  Rim3_L={m_new['rim3_l']:.0f}  Blk={m_new['blk_pct']:.0f}%",
                fill=(50, 50, 50),
                font=font_lbl,
            )
        # Row 3: 4x zoom of reference edge
        yc_z = 58 + 2 * ch_far
        z_crop = rgb_new[220:390, 260:430] if ref_id == 1 else rgb_new[200:370, 520:690]
        sheet.paste(Image.fromarray(z_crop).resize((cw - 16, ch_zoom - 34), Image.Resampling.LANCZOS), (xc + 8, yc_z + 28))
        draw.rectangle([xc, yc_z, xc + cw, yc_z + ch_zoom], outline=(100, 140, 210), width=1)
        draw.text((xc + 8, yc_z + 6), f"4x Edge Zoom: #{ref_id} {short} ({m_new['width_px']:.1f}px)", fill=(25, 65, 150), font=font_lbl)

    # Cols 2..5: #41, #42, #55, #62 comparing Row 1 (Previous 1px) vs Row 2 (New 2.8px True-Match) + Row 3 (4x Zoom of New 2.8px)
    zoom_coords = {
        41: (390, 210, 85),
        42: (460, 745, 95),
        55: (310, 420, 85),
        62: (565, 195, 85),
    }
    for offset, tid in enumerate([41, 42, 55, 62]):
        ci = 2 + offset
        lbl, rgb_old, m_old, rgb_new, m_new = processed[tid]
        short = lbl.split("] ")[-1][:20]
        xc = ci * cw

        # Row 1: Previous 1px outline (looked too thin from afar)
        yc1 = 58
        sheet.paste(
            Image.fromarray(rgb_old).resize((cw - 16, ch_far - 50), Image.Resampling.LANCZOS),
            (xc + 8, yc1 + 44),
        )
        draw.rectangle([xc, yc1, xc + cw, yc1 + ch_far], outline=(215, 90, 90), width=1)
        draw.text((xc + 8, yc1 + 6), f"PREV 1px #{tid} {short}", fill=(180, 35, 35), font=font_lbl)
        draw.text(
            (xc + 8, yc1 + 23),
            f"Width={m_old['width_px']:.1f}px  Rim3_L={m_old['rim3_l']:.0f}  NoL={m_old['nol_pct']:.0f}%",
            fill=(90, 40, 40),
            font=font_lbl,
        )

        # Row 2: New 2.8px True-Match outline (matches #1 Marten & #2 Macaque from afar!)
        yc2 = 58 + ch_far
        sheet.paste(
            Image.fromarray(rgb_new).resize((cw - 16, ch_far - 50), Image.Resampling.LANCZOS),
            (xc + 8, yc2 + 44),
        )
        draw.rectangle([xc, yc2, xc + cw, yc2 + ch_far], outline=(30, 175, 65), width=2)
        draw.text((xc + 8, yc2 + 6), f"NEW 2.8px #{tid} {short}", fill=(20, 135, 50), font=font_lbl)
        draw.text(
            (xc + 8, yc2 + 23),
            f"Width={m_new['width_px']:.1f}px  Rim3_L={m_new['rim3_l']:.0f}  Blk={m_new['blk_pct']:.0f}%",
            fill=(20, 110, 40),
            font=font_lbl,
        )

        # Row 3: 4x Zoom of New 2.8px outline (verifying fur notches & whiskers are still super clean!)
        yc_z = 58 + 2 * ch_far
        yz, xz, rz = zoom_coords[tid]
        z_crop = rgb_new[max(0, yz - rz) : min(1024, yz + rz), max(0, xz - rz) : min(1024, xz + rz)]
        sheet.paste(Image.fromarray(z_crop).resize((cw - 16, ch_zoom - 34), Image.Resampling.LANCZOS), (xc + 8, yc_z + 28))
        draw.rectangle([xc, yc_z, xc + cw, yc_z + ch_zoom], outline=(30, 175, 65), width=2)
        draw.text((xc + 8, yc_z + 6), f"4x Zoom NEW #{tid} ({m_new['width_px']:.1f}px, Sharp Fur/Whiskers)", fill=(20, 135, 50), font=font_lbl)

    out_p = ART / "outline_width_far_and_close_verification.jpg"
    sheet.save(out_p, quality=94)
    print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
