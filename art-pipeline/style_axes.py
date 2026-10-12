"""Empirical style analysis of the master illustrations.

Two kinds of measurement, both computed on the animal only (alpha mask):

1. Classic image analysis (no ML) - each number maps to something an
   illustrator would name:
     outline_px        thickness of the line around the silhouette
     outline_coverage  how much of the silhouette edge actually has a line
     outline_darkness  how dark that line is (1 = pure black)
     outline_tint      how colored the line is (0 = neutral black/gray)
     ink_lines         share of the interior covered by thin dark drawn lines
                       (fur strokes, feather edges, scale outlines)
     flat_areas        share of the interior that is calm, unchanging color
     soft_blend        share that is gradual, smooth tone change
     hard_edges        share that is abrupt tone change
     palette_size      number of distinct colors covering 90% of the animal
     saturation        average color intensity
     contrast          lightest-vs-darkest spread
     shadow_depth      how far shadows drop below the animal's middle tone
     fine_texture      high-frequency energy - tiny repeated detail
     edge_roughness    how broken-up the silhouette is (fur tufts, feathers)
     warmth            red-vs-blue balance

2. A style fingerprint (VGG Gram matrices - the classic "style" representation
   from neural style transfer: it captures brush/texture statistics and
   ignores what the subject is or where it sits in the frame). Used only for
   clustering into style families.

Outputs art-pipeline/vetting/style_axes.json (consumed by axes.html).
Usage: art-pipeline/funnel-venv/Scripts/python art-pipeline/style_axes.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from skimage.morphology import skeletonize
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
MASTERS = ROOT / "masters"
VET = ROOT / "vetting"
SIZE = 768


def load(p: Path):
    im = Image.open(p).convert("RGBA")
    a = np.asarray(im)
    ys, xs = np.where(a[..., 3] > 128)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    im.thumbnail((SIZE, SIZE), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    return a[..., :3], a[..., 3] > 0.5


def classic(rgb, mask) -> dict:
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, val = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    m8 = mask.astype(np.uint8)

    # silhouette edge + outline band
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    edge = mask & ~cv2.erode(m8, k(1)).astype(bool)
    band = mask & ~cv2.erode(m8, k(12)).astype(bool)
    interior = cv2.erode(m8, k(14)).astype(bool)
    if interior.sum() < 200:
        interior = cv2.erode(m8, k(4)).astype(bool)

    med = float(np.median(lum[mask]))
    dark = lum < min(0.32, 0.6 * med)
    dark_band = dark & band
    near_edge_dark = cv2.dilate(dark_band.astype(np.uint8), k(3)).astype(bool)
    coverage = float((near_edge_dark & edge).sum() / max(edge.sum(), 1))
    width = 0.0
    if dark_band.sum() > 50:
        dt = cv2.distanceTransform(dark_band.astype(np.uint8), cv2.DIST_L2, 3)
        sk = skeletonize(dark_band)
        if sk.any():
            width = float(2 * np.median(dt[sk]))
    ol_lum = float(lum[dark_band].mean()) if dark_band.any() else med
    ol_sat = float(sat[dark_band].mean()) if dark_band.any() else 0.0

    # thin dark drawn lines inside: black top-hat picks out strokes narrower
    # than the kernel that are darker than their surroundings
    tophat = cv2.morphologyEx((lum * 255).astype(np.uint8), cv2.MORPH_BLACKHAT, k(3)) / 255.0
    ink_lines = float((tophat[interior] > 0.10).mean())

    # tone transitions
    blur = cv2.GaussianBlur(lum, (0, 0), 1.0)
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    g = np.hypot(gx, gy)[interior]
    flat = float((g < 0.03).mean())
    soft = float(((g >= 0.03) & (g < 0.25)).mean())
    hard = float((g >= 0.25).mean())

    # palette size: distinct quantized colors covering 90% of the animal
    q = (rgb[mask] * 15).round().astype(np.int32)
    codes = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    counts = np.sort(np.bincount(codes))[::-1]
    palette = int(np.searchsorted(np.cumsum(counts), 0.9 * counts.sum()) + 1)

    # fine texture: high-frequency share of the spectrum over the animal bbox
    l2 = np.where(mask, lum, med)
    F = np.abs(np.fft.fftshift(np.fft.fft2(l2 - l2.mean())))
    h, w = F.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    fine = float(F[r > 0.35].sum() / max(F[r > 0.02].sum(), 1e-6))

    # silhouette roughness: perimeter vs perimeter of a smoothed silhouette
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sm = (cv2.GaussianBlur(m8.astype(np.float32), (0, 0), 4) > 0.5).astype(np.uint8)
    cnts_s, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    per_s = sum(cv2.arcLength(c, True) for c in cnts_s)
    rough = float(per / max(per_s, 1.0))

    bright = val > 0.2
    return {
        "outline_px": width, "outline_coverage": coverage,
        "outline_darkness": 1.0 - ol_lum, "outline_tint": ol_sat,
        "ink_lines": ink_lines, "flat_areas": flat, "soft_blend": soft, "hard_edges": hard,
        "palette_size": palette,
        "saturation": float(sat[mask & bright].mean()),
        "contrast": float(np.percentile(lum[mask], 95) - np.percentile(lum[mask], 5)),
        "shadow_depth": float(med - np.percentile(lum[mask], 5)),
        "fine_texture": fine, "edge_roughness": rough,
        "warmth": float((rgb[..., 0] - rgb[..., 2])[mask].mean()),
    }


def gram_fingerprints(items, device):
    vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features.to(device).eval()
    layers = {1, 6, 11, 20}  # relu1_1, relu2_1, relu3_1, relu4_1
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for rgb, mask in items:
        img = np.where(mask[..., None], rgb, 1.0)  # animal on white
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear")
        x = (x - mean) / std
        feats = []
        with torch.no_grad():
            for i, layer in enumerate(vgg):
                x = layer(x)
                if i in layers:
                    f = x[0].flatten(1)
                    g = (f @ f.T) / f.shape[1]
                    feats.append(g[torch.triu_indices(*g.shape, device=device).unbind()].cpu().numpy())
                if i >= max(layers):
                    break
        out.append(np.concatenate([np.log1p(np.abs(v)) * np.sign(v) for v in feats]))
    return np.stack(out)


def main() -> None:
    sets = {n: set(json.loads((VET / f"elite_{n}_v1.json").read_text())["keep"])
            for n in ("ultra", "core", "superset")}
    cls = {s["name"].lower().replace(" ", "_"): s["taxonomy"]["class"]
           for s in json.loads((ROOT.parent / "frontend/public/species_config.json").read_text(encoding="utf-8"))}
    thumbs = VET / "axis_thumbs"
    thumbs.mkdir(exist_ok=True)

    slugs, loaded, rows = [], [], []
    for p in sorted(MASTERS.glob("*.png")):
        rgb, mask = load(p)
        slugs.append(p.stem)
        loaded.append((rgb, mask))
        rows.append(classic(rgb, mask))
        t = thumbs / f"{p.stem}.jpg"
        if not t.exists():
            bg = np.where(mask[..., None], rgb, 1.0)
            im = Image.fromarray((bg * 255).astype(np.uint8))
            im.thumbnail((240, 240), Image.LANCZOS)
            im.save(t, quality=88)
        print(f"measured {p.stem}", flush=True)

    feats = list(rows[0])
    X = StandardScaler().fit_transform(np.array([[r[f] for f in feats] for r in rows]))
    fam_classic = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = gram_fingerprints(loaded, device)
    Gp = PCA(n_components=min(20, len(G) - 1)).fit_transform(StandardScaler().fit_transform(G))
    fam_gram = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(Gp)

    pieces = []
    for i, s in enumerate(slugs):
        tier = ("ultra" if s in sets["ultra"] else "core" if s in sets["core"]
                else "superset" if s in sets["superset"] else "rejected")
        pieces.append({"slug": s, "class": cls.get(s, "?"), "tier": tier,
                       "family_classic": int(fam_classic[i]), "family_gram": int(fam_gram[i]),
                       **{f: round(float(rows[i][f]), 4) for f in feats}})

    # how inconsistent each set is on each axis, relative to the full range
    spread = {}
    for f in feats:
        allv = np.array([p[f] for p in pieces])
        rng = np.percentile(allv, 95) - np.percentile(allv, 5) or 1.0
        spread[f] = {n: round(float((np.percentile(v, 90) - np.percentile(v, 10)) / rng), 3)
                     for n in ("ultra", "core") for v in [np.array([p[f] for p in pieces if p["slug"] in sets[n]])]}
    (VET / "style_axes.json").write_text(json.dumps({"features": feats, "spread": spread, "pieces": pieces}, indent=1))
    print("wrote vetting/style_axes.json")


if __name__ == "__main__":
    main()
