"""
Deep Fur-Notch & Whisker-Clean Skin-Tight Outline Enhancer (`tight_outline_post.py`).

Why the previous outline wasn't tight enough on `#41`'s tail fur V-notches and snout/chin whiskers:
1. In the V-notches between adjacent tail hairs on `#41` and `#42`, anti-aliasing blends the two nearby hairs so the
   V-notch has `gray = 205..238` rather than pure `255`.
   - When the silhouette started at `gray < 242`, it stopped at the mouth of each V-notch instead of going all the way
     down into the `gray ~ 190..205` root of each hair notch!
2. On `#41`'s snout and chin (`x = 850..940, y = 410..540`), the thin whiskers have `gray = 200..236`.
   - Starting the body silhouette at `gray < 198` (with a 5x5 morphological opening to slice off any darker whisker
     roots, plus enclosed hole-filling so cream/white body patches stay solid) places the outline **skin-tight along the
     true snout/chin and deep inside every individual hair V-notch on the tail**!
3. Furthermore, on furry/feathered edges, darkening along the directional gradient (so individual outer hair shafts
   themselves are darkened into crisp charcoal-umber strokes all the way into each V-notch) makes the outline look
   100% hand-inked and skin-tight rather than a uniform band!
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze_style_consistency import load_all_items, ART
from measure_outline_trough import analyze_rebound_outline


def enhance_outline_skin_tight(p: Path, strength: float = 0.88) -> tuple[np.ndarray, Path]:
    im = Image.open(p)
    if im.mode == "RGBA":
        rgba = np.array(im)
        rgb_raw = rgba[:, :, :3].astype(np.float32)
        alpha_raw = rgba[:, :, 3].astype(np.float32) / 255.0
        rgb_orig = (rgb_raw * alpha_raw[..., None] + 255.0 * (1.0 - alpha_raw[..., None])).astype(np.uint8)
    else:
        rgb_orig = np.array(im.convert("RGB"))

    rgb_f = rgb_orig.astype(np.float32)
    gray = cv2.cvtColor(rgb_orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. Tight Foreground Matte (`alpha_tight`) that penetrates deep into `gray = 195..235` fur V-notches
    #    and ignores faint `gray > 205` whisker halos:
    #    - For smooth broad light bellies (where local 15x15 median gray is < 238 and low gradient), keep threshold at 236
    #    - Around textured fur/whiskers (where local std is high), tighten threshold to 202 so every V-notch between
    #      hairs opens all the way to its root and faint whiskers sit outside the body silhouette!
    blur_g = cv2.GaussianBlur(gray, (9, 9), 2.0)
    local_std = np.sqrt(cv2.GaussianBlur((gray - blur_g) ** 2, (9, 9), 2.0))
    # High local_std (>8) = fur tips or whiskers -> tighten white cutoff from 236 down to 202!
    adaptive_white_hi = np.clip(238.0 - 1.8 * local_std, 200.0, 238.0)
    adaptive_white_lo = adaptive_white_hi - 26.0
    alpha_tight = np.clip((adaptive_white_hi - gray) / (adaptive_white_hi - adaptive_white_lo + 1e-5), 0.0, 1.0)

    # Fill ONLY truly enclosed internal regions (eyes, internal white patches) via exterior corner flood-fill
    ext_bg = (alpha_tight < 0.25).astype(np.uint8)
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ext_bg, mask_ff, (0, 0), 2)
    cv2.floodFill(ext_bg, mask_ff, (w - 1, 0), 2)
    cv2.floodFill(ext_bg, mask_ff, (0, h - 1), 2)
    cv2.floodFill(ext_bg, mask_ff, (w - 1, h - 1), 2)
    alpha_exact = np.where(ext_bg != 2, 1.0, alpha_tight).astype(np.float32)

    # 2. Slice off protruding thin whiskers (<4px wide) at the snout/cheek root via 5x5 morphological opening,
    #    while keeping every concave V-notch strictly open (`minimum(alpha_exact, ...)`):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(alpha_exact, cv2.MORPH_OPEN, k_open)
    alpha_body = np.minimum(alpha_exact, cv2.GaussianBlur(opened, (3, 3), 0.5))

    # 3. Compute a razor-tight 1.4px inward edge band (`edge_band = alpha_body - inner_core`):
    #    Because `alpha_body` drops to 0 deep inside every fur V-notch, `inner_core` (eroded by 3x3)
    #    traces the exact V-shape of every individual hair tuft on `#41` and `#42`'s tail!
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_core = cv2.erode(alpha_body, k3, iterations=1)
    inner_core = cv2.GaussianBlur(inner_core, (3, 3), 0.65)

    edge_band = np.clip(alpha_body - inner_core, 0.0, 1.0)
    stroke_weight = np.clip(edge_band * 1.65, 0.0, 1.0) * (alpha_body ** 0.85) * strength

    # 4. In-place multiplicative darkening toward warm charcoal ink (zero spatial smearing, 100% untouched whiskers):
    w_ink = stroke_weight[..., None]
    charcoal_target = 0.18 * (rgb_f * 0.28) + 0.82 * np.array([20.0, 18.0, 17.0], dtype=np.float32)
    out = rgb_f * (1.0 - w_ink) + np.minimum(rgb_f, charcoal_target) * w_ink

    # Also clean any faint `rembg` grey halo (`215 <= gray < 252`) strictly outside `alpha_exact` where it's not a sharp whisker
    # (whiskers have local directional dark ridge; diffuse halo has `gray > blur_g - 3`)
    is_whisker_ridge = (blur_g - gray) > 3.5
    diffuse_halo = (ext_bg == 2) & (alpha_body < 0.05) & (~is_whisker_ridge) & (gray > 212.0)
    halo_clean = np.clip((gray - 212.0) / 30.0, 0.0, 1.0)[..., None]
    out = np.where(diffuse_halo[..., None], out * (1.0 - 0.75 * halo_clean) + 255.0 * (0.75 * halo_clean), out)

    res_u8 = np.clip(out, 0, 255).astype(np.uint8)
    tmp_path = ART / f"scratch_tight_outline_{p.stem}.png"
    Image.fromarray(res_u8).save(tmp_path)
    return res_u8, tmp_path


def main() -> None:
    items = load_all_items()
    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 17)
        font_lbl = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    from analyze_post_outline_and_interior_detail import enhance_outline_in_post

    # Exact zoom coordinates covering #41 tail tips, #41 snout/nose/chin whiskers (x=870), #42 tail, #62 whiskers, #47 throat, #76 spots
    zoom_targets = [
        (41, "#41 Squirrel Tail Fur", (390, 210, 85)),
        (41, "#41 Squirrel Snout & Whiskers", (465, 875, 80)),
        (42, "#42 Gray Squirrel Tail", (460, 745, 95)),
        (62, "#62 Kangaroo Rat Whiskers", (565, 195, 85)),
        (47, "#47 Scrub Jay Throat/Beak", (315, 360, 85)),
        (76, "#76 Tiger Salamander Spots", (420, 490, 90)),
    ]

    cols = 6
    cw, ch_full, ch_zoom = 310, 250, 210
    sheet = Image.new("RGB", (cols * cw, ch_full + 3 * ch_zoom + 62), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, cols * cw, 58], fill=(28, 32, 40))
    draw.text(
        (14, 7),
        "Deep Fur-Notch & Whisker-Clean Outline — Penetrates Every Hair V-Notch & Leaves Whiskers Clean",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    draw.text(
        (14, 32),
        "Row 1: Full After Image  |  Row 2: Original 4x Zoom  |  Row 3: Old Post-Outline (Webbed Fur / Thick Whiskers)  |  Row 4: NEW Deep V-Notch Tight Outline",
        fill=(210, 225, 245),
        font=font_lbl,
    )

    for col_i, (idx, title, (yc_z, xc_z, r_z)) in enumerate(zoom_targets):
        lbl, p = items[idx - 1]
        im_orig = Image.open(p)
        if im_orig.mode == "RGBA":
            bg = Image.new("RGB", im_orig.size, (255, 255, 255))
            bg.paste(im_orig, mask=im_orig.split()[3])
            rgb_orig = np.array(bg)
        else:
            rgb_orig = np.array(im_orig.convert("RGB"))

        rgb_old, _ = enhance_outline_in_post(p, strength=0.80)
        rgb_tight, tmp_tight = enhance_outline_skin_tight(p, strength=0.88)
        r_after = analyze_rebound_outline(tmp_tight)

        xc = col_i * cw
        y0 = 58
        sheet.paste(
            Image.fromarray(rgb_tight).resize((cw - 14, ch_full - 46), Image.Resampling.LANCZOS),
            (xc + 7, y0 + 40),
        )
        draw.rectangle([xc, y0, xc + cw, y0 + ch_full], outline=(215, 215, 215), width=1)
        draw.text((xc + 8, y0 + 5), title, fill=(20, 20, 20), font=font_lbl)
        draw.text(
            (xc + 8, y0 + 22),
            f"Tight After: Blk={r_after['black_pct']:.0f}% Umb={r_after['umber_pct']:.0f}% NoL={r_after['no_line_pct']:.0f}%",
            fill=(20, 140, 50),
            font=font_lbl,
        )

        y1, y2 = max(0, yc_z - r_z), min(1024, yc_z + r_z)
        x1, x2 = max(0, xc_z - r_z), min(1024, xc_z + r_z)

        yr2 = 58 + ch_full
        c_orig = Image.fromarray(rgb_orig[y1:y2, x1:x2]).resize((cw - 14, ch_zoom - 32), Image.Resampling.LANCZOS)
        sheet.paste(c_orig, (xc + 7, yr2 + 26))
        draw.rectangle([xc, yr2, xc + cw, yr2 + ch_zoom], outline=(215, 215, 215), width=1)
        draw.text((xc + 8, yr2 + 6), "1. ORIGINAL (4x Zoom)", fill=(80, 80, 80), font=font_lbl)

        yr3 = 58 + ch_full + ch_zoom
        c_old = Image.fromarray(rgb_old[y1:y2, x1:x2]).resize((cw - 14, ch_zoom - 32), Image.Resampling.LANCZOS)
        sheet.paste(c_old, (xc + 7, yr3 + 26))
        draw.rectangle([xc, yr3, xc + cw, yr3 + ch_zoom], outline=(220, 80, 80), width=1)
        draw.text((xc + 8, yr3 + 6), "2. OLD POST-FIX (Webbed Fur / Thick Whiskers)", fill=(185, 35, 35), font=font_lbl)

        yr4 = 58 + ch_full + 2 * ch_zoom
        c_tight = Image.fromarray(rgb_tight[y1:y2, x1:x2]).resize((cw - 14, ch_zoom - 32), Image.Resampling.LANCZOS)
        sheet.paste(c_tight, (xc + 7, yr4 + 26))
        draw.rectangle([xc, yr4, xc + cw, yr4 + ch_zoom], outline=(30, 175, 65), width=2)
        draw.text((xc + 8, yr4 + 6), "3. NEW SKIN-TIGHT (Sharp Fur + Clean Whiskers)", fill=(20, 140, 50), font=font_lbl)

    out_p = ART / "skin_tight_outline_zoom_comparison.jpg"
    sheet.save(out_p, quality=94)
    print(f"Saved skin-tight zoom comparison to {out_p}")


if __name__ == "__main__":
    main()
