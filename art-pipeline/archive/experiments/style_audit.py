"""Measure objective style properties of every master illustration.

Each metric is computed on the subject only (alpha > 128), at 512px:

  outline_dark     how much darker the silhouette edge is than the interior
                   (0 = no outline, 0.3+ = strong dark contour)
  outline_black    fraction of edge pixels that are near-black (ink-black
                   outline vs tinted/dark-brown outline)
  saturation       mean color saturation of non-dark pixels (muted -> vivid)
  value_range      contrast: 95th - 5th percentile brightness
  ink_fraction     fraction of interior pixels that are near-black
  detail           mean local gradient - how busy the texture is
                   (flat graphic fills = low, every-hair rendering = high)
  flat_fraction    fraction of interior with almost no local change (flat
                   color areas / cel-style fills)
  gloss            fraction of bright, colorless highlight pixels
                   (shiny specular highlights; inflated on white animals)
  warmth           mean red minus blue (warm earth vs cool palette)

Writes art-pipeline/style_metrics.csv. Usage: python art-pipeline/style_audit.py
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).parent
MASTERS = ROOT / "masters"


def metrics(p: Path) -> dict:
    im = Image.open(p).convert("RGBA")
    im.thumbnail((512, 512), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    rgb, alpha = a[..., :3], a[..., 3]
    mask = alpha > 0.5

    m_img = Image.fromarray((mask * 255).astype(np.uint8))
    eroded = np.asarray(m_img.filter(ImageFilter.MinFilter(7))) > 127
    edge = mask & ~eroded
    interior = np.asarray(m_img.filter(ImageFilter.MinFilter(15))) > 127

    v = rgb.max(-1)
    mn = rgb.min(-1)
    s = np.where(v > 1e-6, (v - mn) / np.maximum(v, 1e-6), 0)
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)

    inner = interior if interior.sum() > 50 else mask
    bright_sat = (v > 0.18) & inner
    return {
        "outline_dark": float(lum[inner].mean() - lum[edge].mean()) if edge.any() else 0.0,
        "outline_black": float((lum[edge] < 0.15).mean()) if edge.any() else 0.0,
        "saturation": float(s[bright_sat].mean()) if bright_sat.any() else 0.0,
        "value_range": float(np.percentile(lum[mask], 95) - np.percentile(lum[mask], 5)),
        "ink_fraction": float((lum[inner] < 0.12).mean()),
        "detail": float(grad[inner].mean()),
        "flat_fraction": float((grad[inner] < 0.012).mean()),
        "gloss": float(((v > 0.93) & (s < 0.12) & inner).sum() / max(inner.sum(), 1)),
        "warmth": float((rgb[..., 0] - rgb[..., 2])[mask].mean()),
    }


def main() -> None:
    rows = []
    for p in sorted(MASTERS.glob("*.png")):
        r = {"slug": p.stem, **{k: round(val, 4) for k, val in metrics(p).items()}}
        rows.append(r)
    # tercile labels per metric across all 100, for readable low/mid/high
    keys = [k for k in rows[0] if k != "slug"]
    for k in keys:
        lo, hi = np.percentile([r[k] for r in rows], [33.3, 66.7])
        for r in rows:
            r[f"{k}_band"] = "low" if r[k] <= lo else ("high" if r[k] >= hi else "mid")
    out = ROOT / "style_metrics.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} masters measured -> {out}")


if __name__ == "__main__":
    main()
