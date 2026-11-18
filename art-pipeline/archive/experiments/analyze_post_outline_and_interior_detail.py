"""
1. Tests whether low-outline / umber-outline illustrations can be cleanly fixed in post-processing
   (Adaptive Field-Guide Charcoal-Umber Edge Trough Enhancer) and verifies with the Rebound-Trough detector.
2. Implements a perceptual CIELAB Interior Detail & Texture Measurement System strictly inside the body (dist > 6px),
   invariant to overall animal darkness/lightness:
   - MicroTex (Fine Brush/Fur/Scale Texture, 1-3px scale): mean perceptual delta-E from sigma=2.2px Gaussian smooth
   - PatternRidges% (Distinct Internal Linework, Feather/Scale Edges, Spots & Stripes, 2-6px scale):
     percentage of interior pixels with local DoG perceptual contrast ridge/valley >= 7.5 delta-E
   - SmoothFill% (Smooth Watercolor / Flat Shaded Zones):
     percentage of interior pixels with local 5x5 perceptual delta-E std < 3.0
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze_style_consistency import load_all_items, ART
from measure_outline_trough import PICKED_45, get_halo_free_silhouette, analyze_rebound_outline


def enhance_outline_in_post(p: Path, strength: float = 0.80) -> tuple[np.ndarray, Path]:
    """
    Sub-Pixel Anti-Aliased Field-Guide Charcoal-Umber Outline Enhancer:
    1. Computes the halo-free silhouette at 4x supersampled resolution (4096x4096) with smooth Gaussian
       contour fairing so the boundary has zero 1px stair-stepping even at 3x zoom.
    2. Strips the outer white anti-aliasing halo outside the true silhouette and lays down a smooth,
       tapered warm-charcoal + 20% local-hue ink trough (`d = 0.5..3.0 px`), then downsamples with LANCZOS.
    """
    rgb, solid = get_halo_free_silhouette(p)
    h, w, _ = rgb.shape

    # Upsample solid mask 4x and smooth with Gaussian blur to get a sub-pixel smooth contour without stair-steps
    solid_4x = cv2.resize(solid * 255, (w * 4, h * 4), interpolation=cv2.INTER_LINEAR)
    solid_4x_blur = cv2.GaussianBlur(solid_4x, (15, 15), 3.2)
    # Continuous signed-distance-like field at 1x resolution via supersampled distance transform
    fg_4x = (solid_4x_blur >= 128).astype(np.uint8)
    dist_in_4x = cv2.distanceTransform(fg_4x, cv2.DIST_L2, 5) / 4.0
    dist_out_4x = cv2.distanceTransform(1 - fg_4x, cv2.DIST_L2, 5) / 4.0

    dist_in = cv2.resize(dist_in_4x, (w, h), interpolation=cv2.INTER_AREA)
    dist_out = cv2.resize(dist_out_4x, (w, h), interpolation=cv2.INTER_AREA)

    out = rgb.copy().astype(np.float32)
    # Sample interior color strictly inside the animal (d_in > 3px) so white background never bleeds into local hue
    interior_rgb = cv2.GaussianBlur(out, (15, 15), 4.0)
    ink_rgb = 0.20 * (interior_rgb * 0.35) + 0.80 * np.array([24.0, 21.0, 19.0], dtype=np.float32)

    # Signed distance s: positive inside, negative outside
    s = dist_in - dist_out
    # Smooth brush bell profile centered at s = +1.15px (just inside the edge), width sigma = 0.85px
    stroke_profile = np.exp(-((s - 1.15) ** 2) / (2.0 * (0.85 ** 2)))
    stroke_profile[s < -0.35] = 0.0
    stroke_profile[s > 3.3] = 0.0

    w_ink = np.clip(stroke_profile * strength, 0.0, 0.88)[..., None]
    # Pull any white halo pixels inside the stroke ring down to ink_rgb, and darken existing colored edges
    blended = out * (1.0 - w_ink) + ink_rgb * w_ink
    # Where original pixel was bright halo (L > 180) inside the stroke zone, use blended directly; otherwise minimum
    gray_orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None]
    out = np.where((gray_orig > 175.0) & (w_ink > 0.15), blended, np.minimum(out, blended))

    # Clean sub-pixel anti-aliased transition to pure white (255, 255, 255) outside s < 0.2px
    aa_white = np.clip((0.25 - s) / 0.95, 0.0, 1.0)[..., None]
    out = out * (1.0 - aa_white) + 255.0 * aa_white

    res_u8 = np.clip(out, 0, 255).astype(np.uint8)
    tmp_path = ART / f"scratch_post_outline_{p.stem}.png"
    Image.fromarray(res_u8).save(tmp_path)
    return res_u8, tmp_path


def analyze_interior_detail_cielab(p: Path) -> dict:
    """
    Measures interior detail and texture strictly inside the silhouette (`dist > 6px` from the boundary,
    so the outer outline never contaminates the interior score).
    Uses perceptual CIELAB (L, a, b) so dark animals and light animals are measured on equal footing:
    1. MicroTex (1-3px fine brush/fur/scale texture):
       Mean perceptual delta-E between raw LAB and sigma=2.0px Gaussian-smoothed LAB inside `dist > 6px`.
    2. PatternRidges% (2-6px internal linework, feather/scale seams, spots, stripes):
       Percentage of interior pixels where Multi-Scale Difference-of-Gaussians (sigma=0.8 vs sigma=3.8)
       in CIELAB exceeds 7.0 perceptual delta-E units.
    3. SmoothFill% (flat / smooth watercolor shading zones):
       Percentage of interior pixels where local 5x5 perceptual delta-E deviation is < 3.2.
    4. ColorVar (Medium-scale internal palette richness, sigma=2px vs sigma=12px delta-E):
       Captures multi-tone shading & color transitions across the body.
    """
    rgb, solid = get_halo_free_silhouette(p)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    interior = dist > 6.0
    if interior.sum() < 200:
        interior = dist > 3.0

    # Convert to float32 CIELAB (L in 0..100, a/b in -128..127)
    rgb_f = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2Lab)

    # 1. Fine Micro-Texture (sigma = 2.0px high-pass in perceptual LAB)
    lab_s2 = cv2.GaussianBlur(lab, (0, 0), 2.0)
    de_micro = np.sqrt(np.sum((lab - lab_s2) ** 2, axis=2))

    # 2. Internal Linework / Feather-Scale Ridges / Markings (DoG sigma=0.8 vs sigma=3.8 in LAB)
    lab_s08 = cv2.GaussianBlur(lab, (0, 0), 0.8)
    lab_s38 = cv2.GaussianBlur(lab, (0, 0), 3.8)
    de_dog = np.sqrt(np.sum((lab_s08 - lab_s38) ** 2, axis=2))

    # 3. Medium-Scale Shading & Color Variation (sigma=2.0 vs sigma=12.0 in LAB)
    lab_s12 = cv2.GaussianBlur(lab, (0, 0), 12.0)
    de_med = np.sqrt(np.sum((lab_s2 - lab_s12) ** 2, axis=2))

    vals_micro = de_micro[interior]
    vals_dog = de_dog[interior]
    vals_med = de_med[interior]

    micro_tex = float(np.mean(vals_micro))
    pattern_ridges_pct = float((vals_dog >= 6.5).mean() * 100.0)
    smooth_fill_pct = float((vals_dog < 2.5).mean() * 100.0)
    med_color_var = float(np.mean(vals_med))

    # Combined perceptual detail index (0..100 scale) combining micro-texture + internal linework ridges + medium shading
    detail_score = 4.0 * micro_tex + 0.85 * pattern_ridges_pct + 1.8 * med_color_var

    # Build a visual heatmap of interior detail (Blue = Smooth Watercolor Fill, Green = Medium Shading, Yellow/Red = Crisp Internal Linework/Texture)
    vis_map = np.full_like(rgb, 255)
    norm_dog = np.clip(de_dog / 16.0, 0.0, 1.0)
    cmap_bgr = cv2.applyColorMap((norm_dog * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cmap_rgb = cv2.cvtColor(cmap_bgr, cv2.COLOR_BGR2RGB)
    # Blend 65% heatmap + 35% original image inside interior; keep outer rim light grey so we see silhouette
    edge_band = (dist > 0) & (~interior)
    vis_map[edge_band] = (0.7 * rgb[edge_band] + 0.3 * np.array([200, 200, 200])).astype(np.uint8)
    vis_map[interior] = (0.68 * cmap_rgb[interior] + 0.32 * rgb[interior]).astype(np.uint8)

    return {
        "micro_tex": micro_tex,
        "ridges_pct": pattern_ridges_pct,
        "smooth_pct": smooth_fill_pct,
        "med_var": med_color_var,
        "detail_score": detail_score,
        "vis_map": vis_map,
        "rgb": rgb,
    }


def main() -> None:
    items = load_all_items()
    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 17)
        font_lbl = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    # PART 1: Post-Processing Outline Enhancement Test on 6 Low-Black-Outline Picks
    test_post_ids = [55, 47, 62, 41, 69, 76]
    cw, ch = 310, 290
    zoom_h = 155
    post_sheet = Image.new("RGB", (6 * cw, 2 * ch + zoom_h + 62), (255, 255, 255))
    dp = ImageDraw.Draw(post_sheet)
    dp.rectangle([0, 0, 6 * cw, 58], fill=(30, 34, 42))
    dp.text(
        (14, 7),
        "Can We Fix Missing/Umber Outlines in Post? — Before (Top) vs. Post-Processed Outline (Middle) + 3x Edge Zoom (Bottom)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    dp.text(
        (14, 32),
        "Adaptive Field-Guide Edge Trough Enhancer: Strips outer white AA halo & darkens the d=0.5..3.0px edge trough using 20% local hue + 80% warm charcoal ink",
        fill=(210, 225, 245),
        font=font_lbl,
    )

    print("=== PART 1: POST-PROCESSING OUTLINE FIX VERIFICATION ===")
    for col_i, idx in enumerate(test_post_ids):
        lbl, p = items[idx - 1]
        r_before = analyze_rebound_outline(p)
        post_rgb, tmp_p = enhance_outline_in_post(p, strength=0.80)
        r_after = analyze_rebound_outline(tmp_p)
        short = lbl.split("] ")[-1][:20]
        print(
            f"#{idx:2d} {short:20s} | BEFORE: Blk={r_before['black_pct']:4.1f}% Umb={r_before['umber_pct']:4.1f}% NoL={r_before['no_line_pct']:4.1f}% Line_L={r_before['med_trough_l']:5.1f} "
            f"--> AFTER: Blk={r_after['black_pct']:4.1f}% Umb={r_after['umber_pct']:4.1f}% NoL={r_after['no_line_pct']:4.1f}% Line_L={r_after['med_trough_l']:5.1f}"
        )

        xc = col_i * cw
        # Top row: Before
        yc_top = 58
        post_sheet.paste(
            Image.fromarray(r_before["rgb"]).resize((cw - 14, ch - 48), Image.Resampling.LANCZOS),
            (xc + 7, yc_top + 42),
        )
        dp.rectangle([xc, yc_top, xc + cw, yc_top + ch], outline=(215, 215, 215), width=1)
        dp.text((xc + 8, yc_top + 6), f"BEFORE #{idx} {short}", fill=(180, 40, 40), font=font_lbl)
        dp.text(
            (xc + 8, yc_top + 23),
            f"Blk:{r_before['black_pct']:.0f}% Umb:{r_before['umber_pct']:.0f}% NoL:{r_before['no_line_pct']:.0f}% (L={r_before['med_trough_l']:.0f})",
            fill=(70, 70, 70),
            font=font_lbl,
        )

        # Middle row: After Post Outline
        yc_bot = 58 + ch
        post_sheet.paste(
            Image.fromarray(post_rgb).resize((cw - 14, ch - 48), Image.Resampling.LANCZOS),
            (xc + 7, yc_bot + 42),
        )
        dp.rectangle([xc, yc_bot, xc + cw, yc_bot + ch], outline=(30, 175, 65), width=2)
        dp.text((xc + 8, yc_bot + 6), f"AFTER POST-FIX #{idx} {short}", fill=(20, 140, 50), font=font_lbl)
        dp.text(
            (xc + 8, yc_bot + 23),
            f"Blk:{r_after['black_pct']:.0f}% Umb:{r_after['umber_pct']:.0f}% NoL:{r_after['no_line_pct']:.0f}% (L={r_after['med_trough_l']:.0f})",
            fill=(30, 30, 30),
            font=font_lbl,
        )

        # Bottom row: 3x Zoom side-by-side (Left = Before crop, Right = After crop around top/head/back edge)
        yc_zm = 58 + 2 * ch
        _, solid = get_halo_free_silhouette(p)
        ys_fg, xs_fg = np.where(solid > 0)
        # Pick an upper-middle edge point (around 35% height of bbox) to show head/back edge clearly
        y_target = int(np.percentile(ys_fg, 30))
        row_xs = xs_fg[np.abs(ys_fg - y_target) <= 3]
        x_target = int(np.min(row_xs)) if len(row_xs) > 0 else 512
        r_crop = 75
        y1, y2 = max(0, y_target - r_crop), min(1024, y_target + r_crop)
        x1, x2 = max(0, x_target - r_crop + 25), min(1024, x_target + r_crop + 25)
        crop_b = Image.fromarray(r_before["rgb"][y1:y2, x1:x2]).resize(((cw - 18) // 2, zoom_h - 32), Image.Resampling.LANCZOS)
        crop_a = Image.fromarray(post_rgb[y1:y2, x1:x2]).resize(((cw - 18) // 2, zoom_h - 32), Image.Resampling.LANCZOS)
        post_sheet.paste(crop_b, (xc + 6, yc_zm + 26))
        post_sheet.paste(crop_a, (xc + 12 + (cw - 18) // 2, yc_zm + 26))
        dp.rectangle([xc, yc_zm, xc + cw, yc_zm + zoom_h], outline=(200, 200, 200), width=1)
        dp.text((xc + 8, yc_zm + 6), "3x Edge Zoom:  Before (Left)  vs.  After (Right)", fill=(45, 45, 45), font=font_lbl)

    out_post = ART / "post_outline_fix_comparison.jpg"
    post_sheet.save(out_post, quality=93)
    print(f"Saved post-outline comparison to {out_post}\n")

    # PART 2: Perceptual CIELAB Interior Detail & Texture Analysis across all 45 Picks
    print("=== PART 2: PERCEPTUAL CIELAB INTERIOR DETAIL & TEXTURE (dist > 6px) ===")
    print(
        f"{'ID':>3s} {'Species':34s} | {'MicroTex':>8s} {'Ridges%':>8s} {'MedShade':>8s} {'Smooth%':>8s} {'DetailIdx':>9s}"
    )
    print("-" * 92)
    det_results = {}
    for idx in PICKED_45:
        lbl, p = items[idx - 1]
        d = analyze_interior_detail_cielab(p)
        det_results[idx] = (lbl, d)
        print(
            f"#{idx:2d} {lbl:34s} | {d['micro_tex']:8.2f} {d['ridges_pct']:7.1f}% {d['med_var']:8.2f} "
            f"{d['smooth_pct']:7.1f}% {d['detail_score']:9.1f}"
        )

    # Render a 16-species visual verification board of Interior Detail (4x4 pairs: Original + CIELAB DoG Detail Heatmap)
    demo_det_ids = [17, 9, 22, 1, 15, 6, 11, 14, 30, 41, 47, 55, 57, 66, 68, 76]
    cols_d, rows_d = 4, 4
    dw, dh = 360, 220
    det_sheet = Image.new("RGB", (cols_d * dw, rows_d * dh + 62), (255, 255, 255))
    dd = ImageDraw.Draw(det_sheet)
    dd.rectangle([0, 0, cols_d * dw, 58], fill=(30, 34, 42))
    dd.text(
        (14, 7),
        "Perceptual CIELAB Interior Detail & Texture Detector (Strictly Inside Body dist > 6px, Invariant to Animal Darkness)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    dd.text(
        (14, 32),
        "Heatmap: RED/YELLOW = Internal Linework, Feather/Scale Seams & Markings (Ridges%)  |  GREEN/CYAN = Soft Fur/Shading  |  BLUE = Smooth Watercolor Fill",
        fill=(210, 225, 245),
        font=font_lbl,
    )

    for i, idx in enumerate(demo_det_ids):
        lbl, d = det_results[idx]
        ri, ci = divmod(i, cols_d)
        xc = ci * dw
        yc = 58 + ri * dh
        half_w = (dw - 18) // 2
        img_h = dh - 48
        det_sheet.paste(
            Image.fromarray(d["rgb"]).resize((half_w, img_h), Image.Resampling.LANCZOS),
            (xc + 6, yc + 42),
        )
        det_sheet.paste(
            Image.fromarray(d["vis_map"]).resize((half_w, img_h), Image.Resampling.LANCZOS),
            (xc + 12 + half_w, yc + 42),
        )
        dd.rectangle([xc, yc, xc + dw, yc + dh], outline=(215, 215, 215), width=1)
        short = lbl.split("] ")[-1][:22]
        dd.text((xc + 8, yc + 5), f"#{idx} {short}  (DetailIdx={d['detail_score']:.0f})", fill=(20, 20, 20), font=font_lbl)
        dd.text(
            (xc + 8, yc + 22),
            f"Ridges:{d['ridges_pct']:.0f}%  MicroTex:{d['micro_tex']:.1f}  SmoothFill:{d['smooth_pct']:.0f}%",
            fill=(65, 65, 65),
            font=font_lbl,
        )

    out_det = ART / "interior_detail_cielab_verification.jpg"
    det_sheet.save(out_det, quality=92)
    print(f"Saved interior detail verification sheet to {out_det}")


if __name__ == "__main__":
    main()
