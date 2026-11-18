"""Post-clean (`clean_post_v7`), v3 outline equalizer, outline metrics, comparison boards, and WebP icon export."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_propagation

from animl_art.matte import remove, subject_bbox

_EXCLUDE_PREFIXES = ("icon-",)
_EXCLUDE_MARKERS = ("_old", "_alt")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
        if bold
        else [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
    )
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def load_rgb_flat(p: Path) -> np.ndarray:
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return np.array(bg)
    return np.array(im.convert("RGB"))


def clean_post_v7(
    raw_im: Image.Image,
    cut_im: Image.Image,
    dst_path: Path,
    is_perched: bool = False,
    preserve_thin_appendages: bool = False,
) -> Image.Image:
    """Isolate stylized subject via BiRefNet BEFORE crop/flatten, preserving perches and thin whip tails."""
    cut_rgba = remove(raw_im.convert("RGBA"))
    arr = np.array(cut_rgba)
    alpha = arr[:, :, 3]
    seed_mask = alpha >= 165

    if seed_mask.any():
        low_sup = cv2.morphologyEx((alpha >= 45).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        prop_mask = binary_propagation(seed_mask, structure=np.ones((3, 3), dtype=bool), mask=low_sup)
        arr[~prop_mask, 3] = 0
        boost = prop_mask & (alpha < 220)
        arr[boost, 3] = np.clip(alpha[boost].astype(np.int32) * 3, 185, 255).astype(np.uint8)
    else:
        arr[alpha < 165, 3] = 0

    if is_perched or preserve_thin_appendages:
        cut_arr = np.array(cut_im.convert("RGB"))
        raw_arr = np.array(raw_im.convert("RGB"))
        dark_mask = (cut_arr.min(axis=-1) < 210).astype(np.uint8) * 255
        if preserve_thin_appendages and not is_perched:
            fg_keep = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        else:
            fg_open = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_keep = cv2.erode(fg_open, np.ones((3, 3), np.uint8)) > 0

        cel_cut = cv2.bilateralFilter(cut_arr, d=11, sigmaColor=50, sigmaSpace=50)
        cel_cut = cv2.bilateralFilter(cel_cut, d=11, sigmaColor=50, sigmaSpace=50)

        stripped = fg_keep & (arr[:, :, 3] < 165)
        painted_px = stripped & (raw_arr.mean(axis=-1) <= 225)
        faded_px = stripped & (raw_arr.mean(axis=-1) > 225)
        arr[painted_px, :3] = raw_arr[painted_px]
        arr[faded_px, :3] = cel_cut[faded_px]
        arr[fg_keep, 3] = 255

    min_comp_frac = 0.002 if preserve_thin_appendages else 0.008
    fg_bin = (arr[:, :, 3] >= 165).astype(np.uint8)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(fg_bin)
    if num_l > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = float(areas.max())
        keep_comp = np.zeros_like(fg_bin, dtype=bool)
        for lbl_idx in range(1, num_l):
            if float(stats[lbl_idx, cv2.CC_STAT_AREA]) >= min_comp_frac * max_area:
                keep_comp |= labels == lbl_idx
        arr[~keep_comp, 3] = 0

    cut_clean = Image.fromarray(arr)
    box = subject_bbox(cut_clean)
    if box is None:
        rgb = raw_im.convert("RGB")
        rgb.save(dst_path)
        return rgb

    subj = cut_clean.crop(box)
    max_dim = max(subj.size)
    fill = max_dim / 1024.0
    if 0.74 <= fill <= 0.82:
        subj_final = subj
    else:
        scale = round(1024 * 0.78) / max_dim
        nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
        subj_final = subj.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj_final, ((1024 - subj_final.width) // 2, (1024 - subj_final.height) // 2), subj_final)
    rgb = canvas.convert("RGB")
    rgb.save(dst_path)
    return rgb


def get_halo_free_silhouette(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = load_rgb_flat(p)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg_raw = (gray < 218).astype(np.uint8)
    closed = cv2.morphologyEx(fg_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(fg_raw)
    for c in contours:
        if cv2.contourArea(c) > 400:
            cv2.drawContours(solid, [c], -1, 1, thickness=-1)
    return rgb, solid


def analyze_rebound_outline(p: Path) -> dict:
    """Halo-invariant 1D normal-ray rebound-trough outline solidity detector."""
    rgb, solid = get_halo_free_silhouette(p)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    dist_smooth = cv2.GaussianBlur(dist, (7, 7), 1.5)
    gx = cv2.Sobel(dist_smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist_smooth, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx, ny = gx / mag, gy / mag

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"black_pct": 0.0, "umber_pct": 0.0, "total_pct": 0.0, "no_line_pct": 100.0, "med_rebound": 0.0, "med_trough_l": 128.0}
    pts = np.vstack([c.reshape(-1, 2) for c in contours if len(c) >= 40])[::3]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    d_steps = np.linspace(0.5, 11.5, 23)
    rebounds, trough_ls, cats = [], [], []

    for x0, y0 in pts:
        n_x, n_y = float(nx[y0, x0]), float(ny[y0, x0])
        if abs(n_x) + abs(n_y) < 0.1:
            continue
        xs = np.clip(np.round(x0 + n_x * d_steps).astype(int), 0, w - 1)
        ys = np.clip(np.round(y0 + n_y * d_steps).astype(int), 0, h - 1)
        if dist[ys[-1], xs[-1]] < 3.5:
            continue
        prof = gray[ys, xs]
        idx_min = int(np.argmin(prof[:9]))
        l_trough = float(prof[idx_min])
        start_in = min(len(prof) - 3, idx_min + 4)
        end_in = min(len(prof), idx_min + 15)
        l_body = float(np.percentile(prof[start_in:end_in], 70))
        if l_body < 80.0 and l_trough < 75.0:
            continue
        rebound = l_body - l_trough
        rebounds.append(rebound)
        trough_ls.append(l_trough)
        if rebound >= 18.0 and l_trough < 72.0:
            cats.append(1)
        elif rebound >= 14.0 and l_trough < 152.0:
            cats.append(2)
        else:
            cats.append(3)

    cats_arr = np.array(cats) if cats else np.array([3])
    reb_arr = np.array(rebounds) if rebounds else np.array([0.0])
    tl_arr = np.array(trough_ls) if trough_ls else np.array([128.0])
    black_pct = float((cats_arr == 1).mean() * 100.0)
    umber_pct = float((cats_arr == 2).mean() * 100.0)
    total_pct = black_pct + umber_pct
    outlined = (cats_arr == 1) | (cats_arr == 2)
    return {
        "black_pct": black_pct,
        "umber_pct": umber_pct,
        "total_pct": total_pct,
        "no_line_pct": float((cats_arr == 3).mean() * 100.0),
        "med_rebound": float(np.median(reb_arr)),
        "med_trough_l": float(np.median(tl_arr[outlined])) if outlined.any() else float(np.median(tl_arr)),
    }


def measure_shells_and_width(p: Path) -> dict:
    rgb, solid = get_halo_free_silhouette(p)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    shells = [float(np.median(gray[(dist > d) & (dist <= d + 1)])) if ((dist > d) & (dist <= d + 1)).any() else 255.0 for d in range(5)]
    rim3_mask = (dist > 0.5) & (dist <= 3.0)
    rim3_l = float(np.median(gray[rim3_mask])) if rim3_mask.any() else 255.0
    w_px = 0.0
    for d in np.linspace(0.2, 6.0, 30):
        band = (dist > d) & (dist <= d + 0.4)
        if band.any() and float(np.median(gray[band])) < 85.0:
            w_px += 0.2
    reb = analyze_rebound_outline(p)
    return {"shells": shells, "rim3_l": rim3_l, "width_px": w_px, "blk_pct": reb["black_pct"], "nol_pct": reb["no_line_pct"]}


def measure_aa_midtone_pct(rgb_u8: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    fg = gray < 245
    bg_dil = cv2.dilate((~fg).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0
    boundary = fg & bg_dil
    if not boundary.any():
        return 0.0
    vals = gray[boundary]
    return float(((vals >= 45) & (vals <= 240)).mean() * 100.0)


def measure_trapped_fur_gap(rgb_u8: np.ndarray) -> float:
    rgb = rgb_u8.astype(np.float32)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dw = np.maximum(255.0 - np.min(rgb, axis=2), 255.0 - gray)
    dw[:2, :] = dw[-2:, :] = dw[:, :2] = dw[:, -2:] = 0.0
    ext = (dw < 12.0).astype(np.uint8)
    m = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
    h, w = gray.shape
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ext, m, pt, 2)
    dist = cv2.distanceTransform((ext != 2).astype(np.uint8), cv2.DIST_L2, 5)
    band = (dist >= 2.0) & (dist <= 4.2)
    return float((gray[band] > 185.0).mean() * 100.0) if band.any() else 0.0


def equalize_outline_smooth_tapered(p: Path, strength: float = 0.90) -> tuple[np.ndarray, dict]:
    """v3 fur-fitted, 4x supersampled, curvature-tapered warm-charcoal outline equalizer."""
    rgb_orig = load_rgb_flat(p).astype(np.float32)
    h, w, _ = rgb_orig.shape
    gray_orig = cv2.cvtColor(np.clip(rgb_orig, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    chroma = np.max(rgb_orig, axis=2) - np.min(rgb_orig, axis=2)
    d_white = np.maximum(255.0 - np.min(rgb_orig, axis=2), 255.0 - gray_orig)
    d_white[:2, :] = d_white[-2:, :] = d_white[:, :2] = d_white[:, -2:] = 0.0

    ext_loose = (d_white < 14.0).astype(np.uint8)
    m1 = np.zeros((h + 2, w + 2), np.uint8)
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ext_loose, m1, pt, 2)
    sil_loose = (ext_loose != 2).astype(np.uint8)
    num_cc, labels, stats_cc, _ = cv2.connectedComponentsWithStats(sil_loose)
    for cid in range(1, num_cc):
        if stats_cc[cid, cv2.CC_STAT_AREA] < 120:
            sil_loose[labels == cid] = 0

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
    min_ref = np.where(in_fur_canyon, np.minimum(min_in_fan, cv2.erode(gray_orig, k9)), min_in_fan)

    fg_f = sil_loose.astype(np.float32)
    g_inside_blur = cv2.GaussianBlur(gray_orig * fg_f, (11, 11), 2.5) / np.maximum(0.05, cv2.GaussianBlur(fg_f, (11, 11), 2.5))
    xs_b = np.clip(xx + ux * 5.0, 0, w - 1).astype(np.float32)
    ys_b = np.clip(yy + uy * 5.0, 0, h - 1).astype(np.float32)
    in_b = cv2.remap(fg_f, xs_b, ys_b, cv2.INTER_NEAREST) > 0.5
    b_in = cv2.remap(g_inside_blur, xs_b, ys_b, cv2.INTER_LINEAR)
    true_white_body = in_b & (b_in > 196.0) & (~in_fur_canyon)

    fur_opac = (255.0 - gray_orig) / np.maximum(35.0, 255.0 - min_ref)
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
    ext_a = ((sil_loose == 0) | stage_a).astype(np.uint8)
    ma = np.zeros((h + 2, w + 2), np.uint8)
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ext_a, ma, pt, 2)
    sil_a = (ext_a != 2).astype(np.uint8)

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

    pre_strip = (sil_loose == 1) & (sil_body == 0) & (~whisker_prot)
    near_strip = cv2.dilate(pre_strip.astype(np.uint8), k3) > 0
    sil_body[(sil_body == 1) & (cv2.morphologyEx(sil_body, cv2.MORPH_OPEN, k5) == 0) & near_strip & (gray_orig > 125.0) & (~whisker_prot)] = 0
    sil_body[(sil_body == 1) & (cv2.morphologyEx(sil_body, cv2.MORPH_OPEN, k3) == 0) & near_strip & (gray_orig > 115.0) & (~whisker_prot)] = 0

    n_bc, bc_labels, bc_stats, _ = cv2.connectedComponentsWithStats(sil_body)
    for cid in range(1, n_bc):
        if bc_stats[cid, cv2.CC_STAT_AREA] < 120:
            sil_body[bc_labels == cid] = 0

    stripped_zone = (sil_loose == 1) & (sil_body == 0) & (~whisker_prot)
    adj_stripped = cv2.dilate(stripped_zone.astype(np.uint8), k5) > 0

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
        np.where(sil_body == 1, np.maximum(0.45, alpha_rim), np.where(stripped_zone, 0.0, alpha_std * (cv2.dilate(sil_body, k3) == 1))),
    ).astype(np.float32)

    alpha_4x = np.clip(cv2.resize(alpha_body, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    alpha_4x_smooth = cv2.GaussianBlur(alpha_4x, (5, 5), 0.9)
    dist_4x = cv2.distanceTransform((alpha_4x_smooth >= 0.50).astype(np.uint8), cv2.DIST_L2, 5) / 4.0
    dist_in = cv2.resize(dist_4x, (w, h), interpolation=cv2.INTER_AREA)
    cov_4x = cv2.resize(np.clip((alpha_4x_smooth - 0.12) / 0.76, 0.0, 1.0), (w, h), interpolation=cv2.INTER_AREA)
    alpha_edge_aa = np.where(inner_1px == 1, 1.0, np.where(adj_stripped, cov_4x, np.maximum(alpha_body, cov_4x * 0.85))).astype(np.float32)

    bh5 = cv2.morphologyEx(gray_orig, cv2.MORPH_CLOSE, k5) - gray_orig
    bh7 = cv2.morphologyEx(gray_orig, cv2.MORPH_CLOSE, k7) - gray_orig
    thin_bh = np.maximum(bh5, 0.82 * bh7)
    thin_creases = (dist_in >= 4.0) & (thin_bh > 10.0)
    thin_p5 = float(np.percentile(gray_orig[thin_creases], 5)) if thin_creases.any() else 25.0
    soft_interior = float(np.clip((thin_p5 - 32.0) / 24.0, 0.0, 1.0))

    perim = ((dist_in > 0.0) & (dist_in <= 1.2)).astype(np.float32)
    k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)).astype(np.float32)
    tau = cv2.GaussianBlur(np.clip(cv2.filter2D(perim, cv2.CV_32F, k_disk) / 14.0, 1.0, 2.8), (11, 11), 3.0)
    lap = cv2.Laplacian(cv2.GaussianBlur(dist_in, (7, 7), 1.5), cv2.CV_32F)
    tip_taper_std = np.clip(1.0 + 1.8 * np.minimum(lap + 0.06, 0.0), 0.55, 1.0)
    w_base = np.clip((3.15 / (tau**0.90)) * tip_taper_std, 1.35, 3.15)

    on_wide_plume = cv2.GaussianBlur(cv2.dilate((dist_in >= 5.5).astype(np.uint8), k19).astype(np.float32), (9, 9), 2.0)
    fur_taper_extra = np.clip((tau - 1.36) / 0.46, 0.0, 1.0) * on_wide_plume * (0.35 + 0.65 * soft_interior)
    tip_taper = np.clip(
        1.0 + (1.8 + 0.22 * on_wide_plume * soft_interior) * np.minimum(lap + 0.06, 0.0),
        0.55 - 0.08 * on_wide_plume * soft_interior,
        1.0,
    )
    w_floor = 1.35 - 0.30 * fur_taper_extra
    w_local = np.clip((3.15 / (tau ** (0.90 + 0.25 * fur_taper_extra))) * tip_taper, w_floor, 3.15)

    fur_zone = np.clip((tau - 1.22) / 0.36, 0.0, 1.0) * on_wide_plume
    tip_zone = (1.0 - tip_taper) * np.clip((tau - 1.18) / 0.36, 0.0, 1.0) * on_wide_plume
    ink_harmonize = np.clip(0.58 * soft_interior + 0.42 * fur_zone * soft_interior + 0.18 * fur_zone + 0.15 * tip_zone * soft_interior, 0.0, 0.96)
    local_strength = strength * np.clip(1.0 - 0.10 * soft_interior - 0.16 * fur_zone * soft_interior - 0.06 * tip_zone * soft_interior, 0.70, 1.0)

    p_inward = 1.0 / (1.0 + np.exp(np.clip((dist_in - w_local) / 0.45, -30.0, 30.0)))
    p_inward = np.where(alpha_edge_aa > 0.02, p_inward, 0.0).astype(np.float32)
    whisker_keep = (~whisker_prot).astype(np.float32)
    w_stroke = p_inward * whisker_keep * local_strength

    dist_in_blur = cv2.GaussianBlur(dist_in, (5, 5), 1.0)
    gy_d = cv2.Sobel(dist_in_blur, cv2.CV_32F, 0, 1, ksize=3)
    gx_d = cv2.Sobel(dist_in_blur, cv2.CV_32F, 1, 0, ksize=3)
    gn_d = np.sqrt(gx_d * gx_d + gy_d * gy_d) + np.float32(1e-6)
    xs_f = np.clip(xx + (gx_d / gn_d) * 3.4, 0, w - 1).astype(np.float32)
    ys_f = np.clip(yy + (gy_d / gn_d) * 3.4, 0, h - 1).astype(np.float32)
    c_deep = cv2.remap(rgb_orig, xs_f, ys_f, cv2.INTER_LINEAR)
    c_fur_target = np.where(adj_stripped[..., None], np.minimum(c_deep, cv2.erode(rgb_orig, k7)), c_deep)

    a_safe = np.maximum(alpha_edge_aa, 0.05)[..., None]
    c_fg = np.clip((rgb_orig - 255.0 * (1.0 - alpha_edge_aa[..., None])) / a_safe, 0.0, 255.0)
    g_fg = 0.299 * c_fg[:, :, 0] + 0.587 * c_fg[:, :, 1] + 0.114 * c_fg[:, :, 2]
    g_fur = 0.299 * c_fur_target[:, :, 0] + 0.587 * c_fur_target[:, :, 1] + 0.114 * c_fur_target[:, :, 2]
    pale_excess = np.clip((g_fg - g_fur - 4.0) / 20.0, 0.0, 1.0) * (g_fur < 185.0) * (~true_white_body) * whisker_keep
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

    if soft_interior > 0.02:
        bil = np.clip(rgb_orig, 0, 255).astype(np.uint8)
        for _ in range(3):
            bil = cv2.bilateralFilter(bil, d=9, sigmaColor=48, sigmaSpace=9)
        g_bil = cv2.cvtColor(bil, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx_b = cv2.Sobel(g_bil, cv2.CV_32F, 1, 0, ksize=3)
        gy_b = cv2.Sobel(g_bil, cv2.CV_32F, 0, 1, ksize=3)
        rel_macro_grad = np.sqrt(gx_b * gx_b + gy_b * gy_b) / np.maximum(110.0, cv2.GaussianBlur(g_bil, (15, 15), 3.0)) * 135.0
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
            gxx = cv2.Sobel(gb, cv2.CV_32F, 2, 0, ksize=3) * (sig**2)
            gyy = cv2.Sobel(gb, cv2.CV_32F, 0, 2, ksize=3) * (sig**2)
            gxy = cv2.Sobel(gb, cv2.CV_32F, 1, 1, ksize=3) * (sig**2)
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
            if np.hypot(cc_st[cid, cv2.CC_STAT_WIDTH], cc_st[cid, cv2.CC_STAT_HEIGHT]) >= 17.0 and cc_st[cid, cv2.CC_STAT_AREA] >= 15:
                anat_gate[cc_lbl == cid] = 1.0
        anat_gate = np.clip(cv2.GaussianBlur(anat_gate, (3, 3), 0.7) * 1.35, 0.0, 1.0)

        ridge_prof = np.sqrt(np.clip((thin_bh - 4.5) / 13.5, 0.0, 1.0) * np.clip((vessel - 4.5) / 13.5, 0.0, 1.0))
        w_internal = (
            ridge_prof
            * anat_gate
            * np.maximum(not_fur_swarm, macro_edge * 0.90)
            * plume_damp
            * soft_interior
            * np.clip((gray_orig - 30.0) / 28.0, 0.0, 1.0)
            * np.clip((dist_in - 1.4) / 1.1, 0.0, 1.0)
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


def render_comparison_boards(
    batch_name: str,
    items: list[dict],
    rows: list[dict],
    raw_dir: Path,
    eq_dir: Path,
    out_dir: Path,
    rows_per_board: int | None = None,
) -> list[Path]:
    """Render 4-column visual comparison boards and return saved file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    step = rows_per_board if rows_per_board is not None else (10 if len(items) > 20 else 5)
    font_hdr = load_font(20, bold=True)
    font_sub = load_font(15, bold=True)
    font_sm = load_font(12, bold=False)
    cw, ch, hh = 380, 400, 64
    headers = [
        "1. Selected Ref Photo",
        "2. Input Cutout",
        "3. Raw v7 (Pre-Eq)",
        "4. Final v7 (step 2000) + v3 Eq",
    ]
    saved_paths: list[Path] = []
    row_by_stem = {r["stem"]: r for r in rows}
    valid_items = [it for it in items if it["stem"] in row_by_stem and (eq_dir / f"{it['stem']}.png").exists()]
    for part_idx, start in enumerate(range(0, len(valid_items), step), start=1):
        sub_items = valid_items[start : start + step]
        board = Image.new("RGB", (4 * cw, hh + len(sub_items) * ch), (246, 247, 249))
        draw = ImageDraw.Draw(board)
        draw.rectangle([0, 0, 4 * cw, hh], fill=(24, 29, 38))
        draw.text(
            (14, 7),
            f"animl_style_v7 (step 2000) Batch `{batch_name}` Evaluation Part {part_idx}",
            fill=(255, 255, 255),
            font=font_hdr,
        )
        for ci, htxt in enumerate(headers):
            draw.text((ci * cw + 14, 38), htxt, fill=(196, 212, 232), font=font_sub)

        for ri, it in enumerate(sub_items):
            stem = it["stem"]
            y0 = hh + ri * ch
            m_row = row_by_stem[stem]
            col_paths = [it["photo_p"], it["cut_p"], raw_dir / f"{stem}.png", eq_dir / f"{stem}.png"]
            for ci, cp in enumerate(col_paths):
                x0 = ci * cw
                tile = Image.open(cp).convert("RGB").resize((cw - 16, cw - 16), Image.Resampling.LANCZOS)
                board.paste(tile, (x0 + 8, y0 + 28))
                draw.rectangle([x0, y0, x0 + cw, y0 + ch], outline=(218, 222, 228), width=1)
                if ci == 0:
                    lbl = f"{it['common']} ({it['cls_name']})"
                elif ci == 1:
                    lbl = "Pre-stylization Cutout"
                elif ci == 2:
                    lbl = "Raw v7-2000 (Pre-Equalizer)"
                else:
                    lbl = f"Final v7-2000  Blk={m_row['black_line_pct']:.0f}% AA={m_row['aa_midtone_pct']:.0f}%"
                draw.text((x0 + 10, y0 + 6), lbl[:48], fill=(32, 36, 44), font=font_sm)

        out_p = out_dir / f"v7_{batch_name}_part{part_idx}.jpg"
        board.save(out_p, quality=93)
        saved_paths.append(out_p)
        print(f"Saved comparison board: {out_p}", flush=True)
    return saved_paths


def export_web_icons(icons_dir: Path, out_dir: Path, size: int = 1024, quality: int = 90) -> int:
    """Convert master PNG species illustrations into 1024px WebP assets for Next.js."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_in = total_out = count = 0
    for png in sorted(icons_dir.glob("*.png")):
        if png.name.startswith(_EXCLUDE_PREFIXES) or any(m in png.stem for m in _EXCLUDE_MARKERS):
            continue
        im = Image.open(png).convert("RGBA")
        if max(im.size) > size:
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
        dest = out_dir / (png.stem + ".webp")
        im.save(dest, "WEBP", quality=quality, method=6)
        total_in += png.stat().st_size
        total_out += dest.stat().st_size
        count += 1
    print(f"{count} icons: {total_in / 1e6:.1f} MB PNG -> {total_out / 1e6:.1f} MB WebP")
    return count
