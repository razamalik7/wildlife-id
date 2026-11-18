"""
Outer-vs-Interior Harmonized, Fur-Fitted, 4x Smooth & Curvature-Tapered Outline Equalizer (`smooth_outline_equalizer.py`).

Fixes four visual flaws across all 78 wildlife illustrations:
1. Tightly fits furry & tufted silhouettes onto actual fur blades (eliminating pale matting haze & V-notch gaps):
   - Separates true protruding long whiskers (`>= 9.5px` from `7x7`-opened core body and `p90` radius `<= 1.85px`)
     so short pointed fur blade tips are never truncated by `MORPH_OPEN`.
   - Uses leak-proof directional ray checks along the inward normal (`pass_level`, `rebound`, and
     `min_in_fan` at `0°, ±32°, ±62°`) plus Stage C (`k5`/`k3`) anti-shred cleanup so exterior flood-fill
     penetrates neutral pale haze (`gray > 154`) and inter-blade V-notches right up to fur blade pigment.
2. Eliminates 1px stair-stepping ("jaggies"):
   - Preserves native/local sub-pixel `0..1` anti-aliasing ramps via `4x` supersampled (`4096x4096`)
     continuous Euclidean distance transform, un-premultiplied ink blending, and smooth outer coverage.
3. Eliminates uniform-thickness clumping on zig-zag fur & feathers (Piece 1):
   - Tapers stroke width locally using `13x13` perimeter tortuosity (`tau`) and convex tip curvature
     (`Laplacian(dist)`), adding extra tip delicacy on wide serrated fur plumes (`on_wide_plume`), while
     keeping thin protruding whiskers 100% untouched.
4. Outer-vs-Interior Ink Harmonization (Pieces 2 & 3):
   - Self-calibrates interior thin-crease softness (`soft_interior` from `k5`/`k7` black-hat lines, ignoring
     wide markings) so native charcoal-inked species remain untouched (`soft_interior == 0`), while softly
     painted species receive fur-pigment-tinted charcoal-umber/slate outer ink (`c_ink_fur`) and surgical
     3-pass-bilateral + Hessian-anisotropy + fur-swarm-suppressed internal structural crease darkening.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from analyze_style_consistency import ART, load_all_items
from measure_outline_trough import PICKED_45, analyze_rebound_outline
from verify_width_and_shells import measure_shells_and_width

OUT_78_DIR = Path(__file__).parent / "equalized_candidates_78"
PREV_EQ_DIR = ART / "eq45_tight_v2b"
EQ45_HARMONIZED_V3_DIR = ART / "eq45_harmonized_v3"


def load_rgb_flat(p: Path) -> np.ndarray:
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return np.array(bg)
    return np.array(im.convert("RGB"))


def measure_aa_midtone_pct(rgb_u8: np.ndarray) -> float:
    """
    Measures percentage of 4-connected outer-boundary pixels (L < 245 with at least one 4-neighbor L >= 245)
    whose luminance lies in the smooth anti-aliased transition band `45 <= L <= 240`.
    """
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    fg = gray < 245
    bg = ~fg
    bg_dil = cv2.dilate(bg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0
    boundary = fg & bg_dil
    if not boundary.any():
        return 0.0
    vals = gray[boundary]
    return float(((vals >= 45) & (vals <= 240)).mean() * 100.0)


def measure_trapped_fur_gap(rgb_u8: np.ndarray) -> float:
    """
    Measures percentage of pixels in the `2.0..4.2px` inner rim band that remain bright (`L > 185`)
    immediately inside the outer silhouette.
    """
    rgb = rgb_u8.astype(np.float32)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dw = np.maximum(255.0 - np.min(rgb, axis=2), 255.0 - gray)
    dw[:2, :] = 0.0
    dw[-2:, :] = 0.0
    dw[:, :2] = 0.0
    dw[:, -2:] = 0.0

    ext = (dw < 12.0).astype(np.uint8)
    m = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
    h, w = gray.shape
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ext, m, pt, 2)
    sil = (ext != 2).astype(np.uint8)
    dist = cv2.distanceTransform(sil, cv2.DIST_L2, 5)
    band = (dist >= 2.0) & (dist <= 4.2)
    return float((gray[band] > 185.0).mean() * 100.0) if band.any() else 0.0


def equalize_outline_smooth_tapered(p: Path, strength: float = 0.90) -> tuple[np.ndarray, dict]:
    """
    Applies fur-fitted, 4x-supersampled, native-AA-preserving, curvature-tapered warm-charcoal outline
    equalization to a single wildlife illustration and returns `(res_rgb_u8, debug_stats)`.
    """
    rgb_orig = load_rgb_flat(p).astype(np.float32)
    h, w, _ = rgb_orig.shape
    gray_orig = cv2.cvtColor(np.clip(rgb_orig, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    chroma = np.max(rgb_orig, axis=2) - np.min(rgb_orig, axis=2)
    d_white = np.maximum(255.0 - np.min(rgb_orig, axis=2), 255.0 - gray_orig)
    # Strip any 2px canvas-border compression/cropping noise before flood-filling
    d_white[:2, :] = 0.0
    d_white[-2:, :] = 0.0
    d_white[:, :2] = 0.0
    d_white[:, -2:] = 0.0

    # 1. Loose exterior silhouette via 4-corner flood-fill
    ext_loose = (d_white < 14.0).astype(np.uint8)
    m1 = np.zeros((h + 2, w + 2), np.uint8)
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ext_loose, m1, pt, 2)
    sil_loose = (ext_loose != 2).astype(np.uint8)
    num_cc, labels, stats_cc, _ = cv2.connectedComponentsWithStats(sil_loose)
    for cid in range(1, num_cc):
        if stats_cc[cid, cv2.CC_STAT_AREA] < 120:
            sil_loose[labels == cid] = 0

    # 2. Detect true protruding long whiskers (>=9px from k7 core body & thin <=2.1px p90 radius)
    #    so short pointed fur blade tips are preserved on the silhouette body.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k19 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    core_body = cv2.morphologyEx(sil_loose, cv2.MORPH_OPEN, k7)
    dist_from_core = cv2.distanceTransform(1 - core_body, cv2.DIST_L2, 5)
    dist_loose = cv2.distanceTransform(sil_loose, cv2.DIST_L2, 5)

    thin_parts = ((sil_loose == 1) & (core_body == 0)).astype(np.uint8)
    n_tc, tc_labels, _, _ = cv2.connectedComponentsWithStats(thin_parts)
    whisker_mask = np.zeros((h, w), dtype=bool)
    for cid in range(1, n_tc):
        cc = tc_labels == cid
        if dist_from_core[cc].max() >= 9.5 and np.percentile(dist_loose[cc], 90) <= 1.85:
            whisker_mask |= cc
    whisker_prot = cv2.dilate(whisker_mask.astype(np.uint8), k3) > 0

    # 3. Geodesic grayscale reconstruction (`pass_level`, `rebound`) + inward fan + narrow fur-canyon check
    k8 = np.ones((3, 3), np.uint8)
    pass_level = np.where(sil_loose == 0, 255.0, 0.0).astype(np.float32)
    for _ in range(18):
        pass_level = np.minimum(cv2.dilate(pass_level, k8), gray_orig)
    rebound = gray_orig - pass_level

    dist_blur = cv2.GaussianBlur(dist_loose, (7, 7), 1.4)
    gy = cv2.Sobel(dist_blur, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(dist_blur, cv2.CV_32F, 1, 0, ksize=3)
    gnorm = np.sqrt(gx * gx + gy * gy) + np.float32(1e-6)
    ux = (gx / gnorm).astype(np.float32)
    uy = (gy / gnorm).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    min_in_fan = np.full((h, w), 255.0, dtype=np.float32)
    for deg in [0.0, 32.0, -32.0, 62.0, -62.0]:
        rad = np.deg2rad(deg)
        ca, sa = np.float32(np.cos(rad)), np.float32(np.sin(rad))
        dx, dy = ux * ca - uy * sa, ux * sa + uy * ca
        for s in [1.5, 3.0, 4.5, 6.5]:
            xs_i = np.clip(xx + dx * s, 0, w - 1).astype(np.float32)
            ys_i = np.clip(yy + dy * s, 0, h - 1).astype(np.float32)
            in_sil = cv2.remap(sil_loose.astype(np.float32), xs_i, ys_i, cv2.INTER_NEAREST) > 0.5
            g_in = cv2.remap(gray_orig, xs_i, ys_i, cv2.INTER_LINEAR)
            min_in_fan = np.where(in_sil, np.minimum(min_in_fan, g_in), min_in_fan)

    fur_pigment = ((sil_loose == 1) & (gray_orig < 176.0)).astype(np.uint8)
    in_fur_canyon = cv2.morphologyEx(fur_pigment, cv2.MORPH_CLOSE, k11) > 0
    min_canyon = cv2.erode(gray_orig, k9)
    min_ref = np.where(in_fur_canyon, np.minimum(min_in_fan, min_canyon), min_in_fan)

    # Normalized foreground blur (strictly excludes exterior 255 white background!)
    fg_f = sil_loose.astype(np.float32)
    g_inside_blur = cv2.GaussianBlur(gray_orig * fg_f, (11, 11), 2.5) / np.maximum(
        0.05, cv2.GaussianBlur(fg_f, (11, 11), 2.5)
    )
    xs_b = np.clip(xx + ux * 5.0, 0, w - 1).astype(np.float32)
    ys_b = np.clip(yy + uy * 5.0, 0, h - 1).astype(np.float32)
    in_b = cv2.remap(fg_f, xs_b, ys_b, cv2.INTER_NEAREST) > 0.5
    b_in = cv2.remap(g_inside_blur, xs_b, ys_b, cv2.INTER_LINEAR)
    true_white_body = in_b & (b_in > 196.0) & (~in_fur_canyon)

    # Three-stage fur-blade fitting (`stage_a` + `stage_b` + `stage_c` anti-shred wisp cleanup)
    fur_opac = (255.0 - gray_orig) / np.maximum(35.0, 255.0 - min_ref)

    # Stage A: Tight outer fur-skirt trim (<=4.8px, gray>154) + controlled white V-notch opening (<=9.2px, gray>=200)
    stage_a = (
        (sil_loose == 1)
        & (~whisker_prot)
        & (~true_white_body)
        & (rebound <= 9.0)
        & (chroma < 45.0)
        & (min_ref < 178.0)
        & ((gray_orig - min_ref) > 18.0)
        & (
            ((dist_loose <= 4.8) & (gray_orig > 154.0) & (pass_level >= 152.0) & (fur_opac < 0.63))
            | ((dist_loose <= 9.2) & in_fur_canyon & (gray_orig >= 200.0) & (pass_level >= 198.0) & (fur_opac < 0.54))
        )
    )
    # IMPORTANT: Start from `(sil_loose == 0) | stage_a` so internal white spots are never flooded
    ext_a = ((sil_loose == 0) | stage_a).astype(np.uint8)
    ma = np.zeros((h + 2, w + 2), np.uint8)
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ext_a, ma, pt, 2)
    sil_a = (ext_a != 2).astype(np.uint8)

    # Stage B: 1.8px wall-trim inside newly opened V-notches so diagonal fur walls shed soft 154..200 haze
    dist_a = cv2.distanceTransform(sil_a, cv2.DIST_L2, 5)
    stage_b = (
        (sil_a == 1)
        & (~whisker_prot)
        & (~true_white_body)
        & (dist_a <= 1.8)
        & (dist_loose <= 10.2)
        & (rebound <= 9.5)
        & (gray_orig > 154.0)
        & (chroma < 45.0)
        & (min_ref < 175.0)
        & ((gray_orig - min_ref) > 18.0)
        & (fur_opac < 0.62)
    )
    ext_b = ((sil_a == 0) | stage_b).astype(np.uint8)
    mb = np.zeros((h + 2, w + 2), np.uint8)
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ext_b, mb, pt, 2)
    sil_tight = (ext_b != 2).astype(np.uint8)
    sil_body = ((sil_tight == 1) & (~whisker_mask)).astype(np.uint8)

    # Stage C: Trim newly exposed 1-2px frayed pale hair shreds (`k5`/`k3` opening near stripped fur haze)
    #          so frosted fur tips wrap smoothly around substantial >=4px fur blades while dark claws/toes stay intact
    pre_strip = (sil_loose == 1) & (sil_body == 0) & (~whisker_prot)
    near_strip = cv2.dilate(pre_strip.astype(np.uint8), k3) > 0
    opened_k5 = cv2.morphologyEx(sil_body, cv2.MORPH_OPEN, k5)
    wisp_shreds = (sil_body == 1) & (opened_k5 == 0) & near_strip & (gray_orig > 125.0) & (~whisker_prot)
    sil_body[wisp_shreds] = 0
    opened_k3 = cv2.morphologyEx(sil_body, cv2.MORPH_OPEN, k3)
    stub_shreds = (sil_body == 1) & (opened_k3 == 0) & near_strip & (gray_orig > 115.0) & (~whisker_prot)
    sil_body[stub_shreds] = 0

    n_bc, bc_labels, bc_stats, _ = cv2.connectedComponentsWithStats(sil_body)
    for cid in range(1, n_bc):
        if bc_stats[cid, cv2.CC_STAT_AREA] < 120:
            sil_body[bc_labels == cid] = 0

    stripped_zone = (sil_loose == 1) & (sil_body == 0) & (~whisker_prot)
    adj_stripped = cv2.dilate(stripped_zone.astype(np.uint8), k5) > 0

    # 4. Tight adaptive sub-pixel coverage `alpha_body` on actual fur-blade pigment
    inner_1px = cv2.erode(sil_body, k3, iterations=1)
    d_ref = np.clip(cv2.GaussianBlur(cv2.dilate(d_white * inner_1px, k5), (5, 5), 1.0), 20.0, 245.0)
    alpha_std = np.clip((d_white - 2.0) / (d_ref - 2.0), 0.0, 1.0)
    min_local = cv2.erode(gray_orig, k7)
    fur_hi = np.clip(min_local + 0.42 * (255.0 - min_local), 150.0, 205.0)
    alpha_fur = np.clip((fur_hi - gray_orig) / np.maximum(18.0, fur_hi - min_local), 0.0, 1.0)
    alpha_rim = np.where(adj_stripped, alpha_fur, alpha_std)
    alpha_body = np.where(
        inner_1px == 1,
        1.0,
        np.where(
            sil_body == 1,
            np.maximum(0.45, alpha_rim),
            np.where(stripped_zone, 0.0, alpha_std * (cv2.dilate(sil_body, k3) == 1)),
        ),
    ).astype(np.float32)

    # 5. Crisp 4x supersampled continuous Euclidean inward distance field & outer AA coverage
    alpha_4x = np.clip(cv2.resize(alpha_body, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    alpha_4x_smooth = cv2.GaussianBlur(alpha_4x, (5, 5), 0.9)
    fg_4x = (alpha_4x_smooth >= 0.50).astype(np.uint8)
    dist_4x = cv2.distanceTransform(fg_4x, cv2.DIST_L2, 5) / 4.0
    dist_in = cv2.resize(dist_4x, (w, h), interpolation=cv2.INTER_AREA)
    cov_4x = cv2.resize(np.clip((alpha_4x_smooth - 0.12) / 0.76, 0.0, 1.0), (w, h), interpolation=cv2.INTER_AREA)
    alpha_edge_aa = np.where(
        inner_1px == 1, 1.0, np.where(adj_stripped, cov_4x, np.maximum(alpha_body, cov_4x * 0.85))
    ).astype(np.float32)

    # 6. Self-calibrating interior thin-crease softness (`soft_interior`) + Piece 1–3 harmonization
    bh5 = cv2.morphologyEx(gray_orig, cv2.MORPH_CLOSE, k5) - gray_orig
    bh7 = cv2.morphologyEx(gray_orig, cv2.MORPH_CLOSE, k7) - gray_orig
    thin_bh = np.maximum(bh5, 0.82 * bh7)
    thin_creases = (dist_in >= 4.0) & (thin_bh > 10.0)
    thin_p5 = float(np.percentile(gray_orig[thin_creases], 5)) if thin_creases.any() else 25.0
    soft_interior = float(np.clip((thin_p5 - 32.0) / 24.0, 0.0, 1.0))

    # Piece 1: Curvature-tapered stroke width with wide-plume serrated-fur delicacy
    perim = ((dist_in > 0.0) & (dist_in <= 1.2)).astype(np.float32)
    k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)).astype(np.float32)
    local_perim_sum = cv2.filter2D(perim, cv2.CV_32F, k_disk)
    tau = cv2.GaussianBlur(np.clip(local_perim_sum / 14.0, 1.0, 2.8), (11, 11), 3.0)
    lap = cv2.Laplacian(cv2.GaussianBlur(dist_in, (7, 7), 1.5), cv2.CV_32F)
    tip_taper_std = np.clip(1.0 + 1.8 * np.minimum(lap + 0.06, 0.0), 0.55, 1.0)
    w_base = np.clip((3.15 / (tau ** 0.90)) * tip_taper_std, 1.35, 3.15)

    on_wide_plume = cv2.GaussianBlur(
        cv2.dilate((dist_in >= 5.5).astype(np.uint8), k19).astype(np.float32), (9, 9), 2.0
    )
    fur_taper_extra = np.clip((tau - 1.36) / 0.46, 0.0, 1.0) * on_wide_plume * (0.35 + 0.65 * soft_interior)
    tip_taper = np.clip(
        1.0 + (1.8 + 0.22 * on_wide_plume * soft_interior) * np.minimum(lap + 0.06, 0.0),
        0.55 - 0.08 * on_wide_plume * soft_interior,
        1.0,
    )
    w_floor = 1.35 - 0.30 * fur_taper_extra
    w_local = np.clip((3.15 / (tau ** (0.90 + 0.25 * fur_taper_extra))) * tip_taper, w_floor, 3.15)

    # Piece 2: Fur-pigment-tinted charcoal-umber/slate outer ink (`c_ink_fur`)
    fur_zone = np.clip((tau - 1.22) / 0.36, 0.0, 1.0) * on_wide_plume
    tip_zone = (1.0 - tip_taper) * np.clip((tau - 1.18) / 0.36, 0.0, 1.0) * on_wide_plume
    ink_harmonize = np.clip(
        0.58 * soft_interior
        + 0.42 * fur_zone * soft_interior
        + 0.18 * fur_zone
        + 0.15 * tip_zone * soft_interior,
        0.0,
        0.96,
    )
    local_strength = strength * np.clip(
        1.0 - 0.10 * soft_interior - 0.16 * fur_zone * soft_interior - 0.06 * tip_zone * soft_interior,
        0.70,
        1.0,
    )

    p_inward = 1.0 / (1.0 + np.exp(np.clip((dist_in - w_local) / 0.45, -30.0, 30.0)))
    p_inward = np.where(alpha_edge_aa > 0.02, p_inward, 0.0).astype(np.float32)
    whisker_keep = (~whisker_prot).astype(np.float32)
    w_stroke = p_inward * whisker_keep * local_strength

    # 7. Stronger inward fur-pigment de-haloing (`3.4px` inward + `k7` erode, anchored to `w_base + 2.4`)
    dist_in_blur = cv2.GaussianBlur(dist_in, (5, 5), 1.0)
    gy_d = cv2.Sobel(dist_in_blur, cv2.CV_32F, 0, 1, ksize=3)
    gx_d = cv2.Sobel(dist_in_blur, cv2.CV_32F, 1, 0, ksize=3)
    gn_d = np.sqrt(gx_d * gx_d + gy_d * gy_d) + np.float32(1e-6)
    ux_d = (gx_d / gn_d).astype(np.float32)
    uy_d = (gy_d / gn_d).astype(np.float32)
    xs_f = np.clip(xx + ux_d * 3.4, 0, w - 1).astype(np.float32)
    ys_f = np.clip(yy + uy_d * 3.4, 0, h - 1).astype(np.float32)
    c_deep = cv2.remap(rgb_orig, xs_f, ys_f, cv2.INTER_LINEAR)
    c_erode_fur = cv2.erode(rgb_orig, k7)
    c_fur_target = np.where(adj_stripped[..., None], np.minimum(c_deep, c_erode_fur), c_deep)

    a_safe = np.maximum(alpha_edge_aa, 0.05)[..., None]
    c_fg = np.clip((rgb_orig - 255.0 * (1.0 - alpha_edge_aa[..., None])) / a_safe, 0.0, 255.0)
    g_fg = 0.299 * c_fg[:, :, 0] + 0.587 * c_fg[:, :, 1] + 0.114 * c_fg[:, :, 2]
    g_fur = 0.299 * c_fur_target[:, :, 0] + 0.587 * c_fur_target[:, :, 1] + 0.114 * c_fur_target[:, :, 2]
    pale_excess = (
        np.clip((g_fg - g_fur - 4.0) / 20.0, 0.0, 1.0)
        * (g_fur < 185.0)
        * (~true_white_body)
        * whisker_keep
    )
    p_dehalo = (1.0 / (1.0 + np.exp(np.clip((dist_in - (w_base + 2.4)) / 0.60, -30.0, 30.0)))) * pale_excess
    c_fg = c_fg * (1.0 - p_dehalo[..., None]) + np.minimum(c_fg, c_fur_target) * p_dehalo[..., None]

    c_ink_solid = 0.14 * (c_fg * 0.22) + 0.86 * np.array([18.0, 16.0, 15.0], dtype=np.float32)
    c_ink_fur = 0.68 * np.clip(c_fur_target * 0.46, 33.0, 52.0) + 0.32 * np.array([35.0, 31.0, 29.0], dtype=np.float32)
    c_ink = c_ink_solid * (1.0 - ink_harmonize[..., None]) + c_ink_fur * ink_harmonize[..., None]

    c_fg_dark = c_fg * (1.0 - w_stroke[..., None]) + np.minimum(c_fg, c_ink) * w_stroke[..., None]
    c_recomp = alpha_edge_aa[..., None] * c_fg_dark + (1.0 - alpha_edge_aa[..., None]) * 255.0

    rim_blend = np.clip(np.maximum(w_stroke, p_dehalo) / 0.12, 0.0, 1.0)[..., None] * whisker_keep[..., None]
    out_dark = rgb_orig * (1.0 - rim_blend) + np.minimum(rgb_orig, c_recomp) * rim_blend
    clean_mask = (cv2.dilate(stripped_zone.astype(np.uint8), k3) > 0) & (~whisker_prot)
    out = np.where(clean_mask[..., None], c_recomp, out_dark)

    # 8. Piece 3: Surgical internal structural crease darkening (`w_internal`) when `soft_interior > 0.02`
    if soft_interior > 0.02:
        rgb_u8 = np.clip(rgb_orig, 0, 255).astype(np.uint8)
        bil = rgb_u8.copy()
        for _ in range(3):
            bil = cv2.bilateralFilter(bil, d=9, sigmaColor=48, sigmaSpace=9)
        g_bil = cv2.cvtColor(bil, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx_b = cv2.Sobel(g_bil, cv2.CV_32F, 1, 0, ksize=3)
        gy_b = cv2.Sobel(g_bil, cv2.CV_32F, 0, 1, ksize=3)
        rel_macro_grad = (
            np.sqrt(gx_b * gx_b + gy_b * gy_b)
            / np.maximum(110.0, cv2.GaussianBlur(g_bil, (15, 15), 3.0))
            * 135.0
        )
        macro_edge = cv2.GaussianBlur(np.clip((rel_macro_grad - 26.0) / 38.0, 0.0, 1.0), (5, 5), 1.2)

        bh15 = cv2.morphologyEx(gray_orig, cv2.MORPH_CLOSE, k15) - gray_orig
        narrow_ok = (thin_bh / np.maximum(bh15, 5.8)) > 0.46

        swarm_density = cv2.GaussianBlur((thin_bh > 8.5).astype(np.float32), (27, 27), 7.0)
        not_fur_swarm = np.clip(1.0 - (swarm_density - 0.14) / 0.14, 0.0, 1.0)
        structural_ok = (not_fur_swarm > 0.48) | (macro_edge > 0.58)
        plume_damp = np.where((swarm_density > 0.16) & (macro_edge < 0.62), 0.45, 1.0).astype(np.float32)

        vessel = np.zeros((h, w), dtype=np.float32)
        for sig in (0.85, 1.38):
            ksz = int(round(sig * 6)) | 1
            gb = cv2.GaussianBlur(gray_orig, (ksz, ksz), sig)
            gxx = cv2.Sobel(gb, cv2.CV_32F, 2, 0, ksize=3) * (sig ** 2)
            gyy = cv2.Sobel(gb, cv2.CV_32F, 0, 2, ksize=3) * (sig ** 2)
            gxy = cv2.Sobel(gb, cv2.CV_32F, 1, 1, ksize=3) * (sig ** 2)
            tr = gxx + gyy
            det_d = np.sqrt(np.maximum(0.0, (gxx - gyy) ** 2 + 4.0 * gxy * gxy))
            lam1 = 0.5 * (tr + det_d)
            lam2 = 0.5 * (tr - det_d)
            aniso = np.clip(1.0 - np.abs(lam2) / (np.maximum(lam1, 1e-4) * 0.57), 0.0, 1.0)
            vessel = np.maximum(vessel, np.maximum(0.0, lam1) * aniso)

        cand_crease = (
            (dist_in >= 2.0)
            & (thin_bh > 8.0)
            & narrow_ok
            & structural_ok
            & (vessel > 7.5)
            & (macro_edge > 0.28)
            & (gray_orig < 148.0)
        )
        n_cc, cc_lbl, cc_st, _ = cv2.connectedComponentsWithStats(cand_crease.astype(np.uint8), connectivity=8)
        anat_gate = np.zeros((h, w), dtype=np.float32)
        for cid in range(1, n_cc):
            span = np.hypot(cc_st[cid, cv2.CC_STAT_WIDTH], cc_st[cid, cv2.CC_STAT_HEIGHT])
            if span >= 17.0 and cc_st[cid, cv2.CC_STAT_AREA] >= 15:
                anat_gate[cc_lbl == cid] = 1.0
        anat_gate = np.clip(cv2.GaussianBlur(anat_gate, (3, 3), 0.7) * 1.35, 0.0, 1.0)

        ridge_prof = np.sqrt(np.clip((thin_bh - 4.5) / 13.5, 0.0, 1.0) * np.clip((vessel - 4.5) / 13.5, 0.0, 1.0))
        already_dark_guard = np.clip((gray_orig - 30.0) / 28.0, 0.0, 1.0)
        in_handoff = np.clip((dist_in - 1.4) / 1.1, 0.0, 1.0)
        w_internal = (
            ridge_prof
            * anat_gate
            * np.maximum(not_fur_swarm, macro_edge * 0.90)
            * plume_damp
            * soft_interior
            * already_dark_guard
            * in_handoff
            * 0.88
        )
        c_int_ink = 0.52 * np.clip(rgb_orig * 0.36, 20.0, 48.0) + 0.48 * np.array([28.0, 25.0, 23.0], dtype=np.float32)
        out = out * (1.0 - w_internal[..., None]) + np.minimum(out, c_int_ink) * w_internal[..., None]
    else:
        w_internal = np.zeros((h, w), dtype=np.float32)

    res_u8 = np.clip(out, 0, 255).astype(np.uint8)
    perim_mask = perim > 0.5
    stats = {
        "mean_tau": float(np.mean(tau[perim_mask])) if perim_mask.any() else 1.0,
        "mean_w_local": float(np.mean(w_local[perim_mask])) if perim_mask.any() else 2.25,
        "whisker_px": int(whisker_mask.sum()),
        "stripped_px": int(stripped_zone.sum()),
        "soft_interior": soft_interior,
        "int_crease_px": int((w_internal > 0.22).sum()),
    }
    return res_u8, stats


def run_verification(items: list[tuple[str, Path]]) -> None:
    test_ids = [1, 2, 4, 9, 15, 41, 42, 43, 47, 55, 62, 68]
    tmp_dir = ART / "scratch_smooth_verify"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prev_eq_dir = PREV_EQ_DIR

    print("=== STEP 2: QUANTITATIVE AA, FUR-GAP, HARMONIZATION & OUTLINE VERIFICATION (12 TEST SPECIES) ===")
    print(
        f"{'ID':>3s} {'Species':30s} | {'Soft':>4s} {'IntPx':>5s} | {'PrevAA':>6s} {'NewAA':>6s} | "
        f"{'PrevGap':>7s} {'NewGap':>7s} | {'Blk%':>6s} {'Tot%':>6s} {'W':>5s} {'RimL':>5s}"
    )
    print("-" * 121)

    new_imgs = {}
    new_metrics = {}
    for idx in test_ids:
        lbl, p = items[idx - 1]
        prev_files = sorted(prev_eq_dir.glob(f"cand_{idx:02d}_*.png")) if prev_eq_dir.exists() else []
        prev_rgb = load_rgb_flat(prev_files[0]) if prev_files else load_rgb_flat(p)
        prev_aa = measure_aa_midtone_pct(prev_rgb)
        prev_gap = measure_trapped_fur_gap(prev_rgb)

        new_rgb, dbg = equalize_outline_smooth_tapered(p)
        new_aa = measure_aa_midtone_pct(new_rgb)
        new_gap = measure_trapped_fur_gap(new_rgb)

        out_p = tmp_dir / f"cand_{idx:02d}_{p.stem}.png"
        Image.fromarray(new_rgb).save(out_p)

        reb = analyze_rebound_outline(out_p)
        sh = measure_shells_and_width(out_p)
        new_imgs[idx] = (lbl, prev_rgb, new_rgb)
        new_metrics[idx] = (prev_aa, new_aa, prev_gap, new_gap, dbg, reb, sh)

        short = lbl.split("] ")[-1][:30]
        print(
            f"#{idx:2d} {short:30s} | {dbg['soft_interior']:4.2f} {dbg['int_crease_px']:5d} | "
            f"{prev_aa:5.1f}% {new_aa:5.1f}% | {prev_gap:6.1f}% {new_gap:6.1f}% | "
            f"{reb['black_pct']:5.1f}% {reb['total_pct']:5.1f}% {sh['width_px']:4.1f} {sh['rim3_l']:5.1f}"
        )

    # Render `harmonized_v3_verification.jpg` (3 rows x 7 cols, anatomy-focused LANCZOS crops)
    zoom_specs = [
        (1, "#1 Marten Head & Fur", (315, 335, 115)),
        (41, "#41 Squirrel Face/Ear/Leg", (485, 775, 120)),
        (41, "#41 Squirrel Fluffy Tail", (360, 255, 135)),
        (42, "#42 Squirrel Face/Ear/Paws", (540, 205, 115)),
        (42, "#42 Squirrel Arched Tail", (445, 665, 180)),
        (43, "#43 Marmot Head & Chest", (315, 765, 125)),
        (15, "#15 Mountain Goat (Guard)", (340, 320, 130)),
    ]

    cols = 7
    cw, ch_far, ch_zm = 270, 265, 275
    sheet = Image.new("RGB", (cols * cw, ch_far + 2 * ch_zm + 64), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 17)
        font_lbl = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    draw.rectangle([0, 0, cols * cw, 60], fill=(24, 29, 38))
    draw.text(
        (14, 7),
        "Outer-vs-Interior Ink & Serrated-Fur Harmonization Check (v3 — Face, Paws, Tail & Coat)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    draw.text(
        (14, 33),
        "Row 1: NEW Harmonized v3 Full Far View  |  Row 2: PREV Tight v2b (eq45_tight_v2b)  |  Row 3: NEW Harmonized v3 Crop",
        fill=(205, 225, 245),
        font=font_lbl,
    )

    for ci, (idx, title, (yc_z, xc_z, r_z)) in enumerate(zoom_specs):
        _, prev_rgb, new_rgb = new_imgs[idx]
        prev_aa, new_aa, prev_gap, new_gap, dbg, reb, sh = new_metrics[idx]
        xc = ci * cw

        # Row 1: Far view
        y1_row = 60
        sheet.paste(Image.fromarray(new_rgb).resize((cw - 14, ch_far - 48), Image.Resampling.LANCZOS), (xc + 7, y1_row + 42))
        draw.rectangle([xc, y1_row, xc + cw, y1_row + ch_far], outline=(210, 215, 225), width=1)
        draw.text((xc + 7, y1_row + 6), title, fill=(15, 20, 30), font=font_lbl)
        draw.text(
            (xc + 7, y1_row + 22),
            f"Blk:{reb['black_pct']:.0f}% Soft:{dbg['soft_interior']:.2f} Int:{dbg['int_crease_px']}",
            fill=(25, 110, 55),
            font=font_lbl,
        )

        ya, yb = max(0, yc_z - r_z), min(1024, yc_z + r_z)
        xa, xb = max(0, xc_z - r_z), min(1024, xc_z + r_z)
        zm_box = (cw - 14, ch_zm - 36)

        # Row 2: Previous Tight v2b eq45_tight_v2b anatomy crop
        y2_row = 60 + ch_far
        crop_prev = Image.fromarray(prev_rgb[ya:yb, xa:xb]).resize(zm_box, Image.Resampling.LANCZOS)
        sheet.paste(crop_prev, (xc + 7, y2_row + 28))
        draw.rectangle([xc, y2_row, xc + cw, y2_row + ch_zm], outline=(220, 120, 50), width=2)
        draw.text(
            (xc + 7, y2_row + 7),
            f"PREV v2b (AA={prev_aa:.0f}% Gap={prev_gap:.1f}%)",
            fill=(175, 70, 20),
            font=font_lbl,
        )

        # Row 3: New Harmonized v3 anatomy crop
        y3_row = 60 + ch_far + ch_zm
        crop_new = Image.fromarray(new_rgb[ya:yb, xa:xb]).resize(zm_box, Image.Resampling.LANCZOS)
        sheet.paste(crop_new, (xc + 7, y3_row + 28))
        draw.rectangle([xc, y3_row, xc + cw, y3_row + ch_zm], outline=(30, 175, 65), width=2)
        draw.text(
            (xc + 7, y3_row + 7),
            f"NEW Harmonized v3 (AA={new_aa:.0f}% Gap={new_gap:.1f}%)",
            fill=(20, 135, 50),
            font=font_lbl,
        )

    out_sheet = ART / "harmonized_v3_verification.jpg"
    sheet.save(out_sheet, quality=94)
    print(f"Saved {out_sheet}")


def run_all_78(items: list[tuple[str, Path]]) -> None:
    OUT_78_DIR.mkdir(parents=True, exist_ok=True)
    EQ45_HARMONIZED_V3_DIR.mkdir(parents=True, exist_ok=True)
    picked_set = set(PICKED_45)

    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 18)
        font_sub = ImageFont.truetype("arialbd.ttf", 13)
        font_lbl = ImageFont.truetype("arialbd.ttf", 11)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    records = []
    print("=== STEP 3: EQUALIZING ALL 78 CANDIDATES WITH HARMONIZED V3 EQUALIZER ===")
    print(
        f"{'ID':>3s} {'Pick':>4s} {'Species':34s} | {'Soft':>4s} {'IntPx':>5s} {'AA%':>6s} {'Gap%':>6s} {'tau':>5s} | {'Blk%':>6s} {'Tot%':>6s} {'W':>5s} {'RimL':>5s}"
    )
    print("-" * 114)

    for idx_0, (lbl, p) in enumerate(items):
        idx = idx_0 + 1
        eq_rgb, dbg = equalize_outline_smooth_tapered(p)
        eq_path = OUT_78_DIR / f"cand_{idx:02d}_{p.stem}.png"
        Image.fromarray(eq_rgb).save(eq_path)

        is_picked = idx in picked_set
        if is_picked:
            shutil.copy2(eq_path, EQ45_HARMONIZED_V3_DIR / eq_path.name)

        aa_pct = measure_aa_midtone_pct(eq_rgb)
        gap_pct = measure_trapped_fur_gap(eq_rgb)
        reb = analyze_rebound_outline(eq_path)
        sh = measure_shells_and_width(eq_path)

        mark = "[*]" if is_picked else "   "
        print(
            f"#{idx:2d} {mark:4s} {lbl:34s} | {dbg['soft_interior']:4.2f} {dbg['int_crease_px']:5d} {aa_pct:5.1f}% {gap_pct:5.1f}% {dbg['mean_tau']:5.2f} | "
            f"{reb['black_pct']:5.1f}% {reb['total_pct']:5.1f}% {sh['width_px']:4.1f} {sh['rim3_l']:5.1f}"
        )
        records.append((idx, lbl, eq_rgb, aa_pct, gap_pct, dbg, reb, sh, is_picked))

    picked_recs = [r for r in records if r[8]]
    print("\n=== 45 PICKED SET SUMMARY (HARMONIZED V3 EQUALIZATION) ===")
    print(f"AA_Midtone%   : mean={np.mean([r[3] for r in picked_recs]):.1f}% (min={np.min([r[3] for r in picked_recs]):.1f}%)")
    print(f"TrappedFurGap%: mean={np.mean([r[4] for r in picked_recs]):.1f}% (max={np.max([r[4] for r in picked_recs]):.1f}%)")
    print(f"BlackLine%    : mean={np.mean([r[6]['black_pct'] for r in picked_recs]):.1f}% (min={np.min([r[6]['black_pct'] for r in picked_recs]):.1f}%)")
    print(f"TotalLine%    : mean={np.mean([r[6]['total_pct'] for r in picked_recs]):.1f}% (min={np.min([r[6]['total_pct'] for r in picked_recs]):.1f}%)")
    print(f"StrokeWidth   : mean={np.mean([r[7]['width_px'] for r in picked_recs]):.1f}px (std={np.std([r[7]['width_px'] for r in picked_recs]):.1f}px)")
    print(f"Rim3_L        : mean={np.mean([r[7]['rim3_l'] for r in picked_recs]):.1f} (max={np.max([r[7]['rim3_l'] for r in picked_recs]):.1f})")

    # Render `picked_45_harmonized_v3_board.jpg`
    cols45, rows45 = 9, 5
    tw, th = 225, 240
    board45 = Image.new("RGB", (cols45 * tw, rows45 * th + 60), (255, 255, 255))
    d45 = ImageDraw.Draw(board45)
    d45.rectangle([0, 0, cols45 * tw, 56], fill=(24, 29, 38))
    d45.text(
        (14, 7),
        "Your 45 Picked Set — Harmonized Outer/Interior Ink, Tight Fur-Fitted & Smooth Outlines (v3)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    d45.text(
        (14, 32),
        f"Mean AA_Midtone={np.mean([r[3] for r in picked_recs]):.1f}%  |  Mean TrappedGap={np.mean([r[4] for r in picked_recs]):.1f}%  |  Mean BlackLine={np.mean([r[6]['black_pct'] for r in picked_recs]):.1f}%  |  Mean Rim3_L={np.mean([r[7]['rim3_l'] for r in picked_recs]):.1f}",
        fill=(200, 235, 210),
        font=font_sub,
    )

    for k, (idx, lbl, eq_rgb, aa_pct, gap_pct, dbg, reb, sh, _) in enumerate(picked_recs):
        ri, ci = divmod(k, cols45)
        xc, yc = ci * tw, 56 + ri * th
        board45.paste(Image.fromarray(eq_rgb).resize((tw - 12, th - 48), Image.Resampling.LANCZOS), (xc + 6, yc + 42))
        d45.rectangle([xc + 1, yc + 1, xc + tw - 2, yc + th - 2], outline=(35, 175, 70), width=2)
        short = lbl.split("] ")[-1][:20]
        d45.text((xc + 6, yc + 5), f"#{idx} {short}", fill=(15, 15, 15), font=font_lbl)
        d45.text(
            (xc + 6, yc + 21),
            f"Blk:{reb['black_pct']:.0f}% AA:{aa_pct:.0f}% Gap:{gap_pct:.1f}%",
            fill=(25, 125, 50),
            font=font_lbl,
        )

    out45 = ART / "picked_45_harmonized_v3_board.jpg"
    board45.save(out45, quality=93)
    print(f"Saved {out45}")

    # Render `all_78_harmonized_v3_board.jpg`
    cols78, rows78 = 10, 8
    tw78, th78 = 210, 225
    board78 = Image.new("RGB", (cols78 * tw78, rows78 * th78 + 60), (255, 255, 255))
    d78 = ImageDraw.Draw(board78)
    d78.rectangle([0, 0, cols78 * tw78, 56], fill=(24, 29, 38))
    d78.text(
        (14, 7),
        "All 78 Candidates (#1-#78) — Harmonized Outer/Interior Ink, Tight Fur-Fitted & Smooth Outlines (v3)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    d78.text(
        (14, 32),
        "GREEN BORDER = In Your 45 Picks  |  GREY BORDER = Other 33 Candidates (All Uniformly Harmonized with Tight Fur Fit & Curvature Taper)",
        fill=(210, 225, 245),
        font=font_sub,
    )

    for k, (idx, lbl, eq_rgb, aa_pct, gap_pct, dbg, reb, sh, is_picked) in enumerate(records):
        ri, ci = divmod(k, cols78)
        xc, yc = ci * tw78, 56 + ri * th78
        if not is_picked:
            d78.rectangle([xc, yc, xc + tw78, yc + th78], fill=(249, 250, 252))
        board78.paste(Image.fromarray(eq_rgb).resize((tw78 - 12, th78 - 46), Image.Resampling.LANCZOS), (xc + 6, yc + 40))
        bcol, bw, tcol = ((30, 180, 65), 3, (15, 115, 40)) if is_picked else ((195, 200, 208), 1, (90, 95, 105))
        d78.rectangle([xc + 1, yc + 1, xc + tw78 - 2, yc + th78 - 2], outline=bcol, width=bw)
        short = lbl.split("] ")[-1][:18]
        d78.text((xc + 6, yc + 5), f"#{idx} {short}", fill=tcol if is_picked else (40, 40, 40), font=font_lbl)
        d78.text(
            (xc + 6, yc + 20),
            f"Blk:{reb['black_pct']:.0f}% AA:{aa_pct:.0f}% Gap:{gap_pct:.1f}%",
            fill=tcol,
            font=font_lbl,
        )

    out78 = ART / "all_78_harmonized_v3_board.jpg"
    board78.save(out78, quality=92)
    print(f"Saved {out78}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    items = load_all_items()
    if args.verify_only or not args.all:
        run_verification(items)
    if args.all:
        run_all_78(items)


if __name__ == "__main__":
    main()
