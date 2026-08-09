"""Extract the working palette of the existing icon set via k-means.

Samples opaque pixels from every species illustration and clusters them.
Output feeds the style bible: real hex values instead of a model's eyeball.

Usage:
    python art-pipeline/extract_palette.py [--icons DIR] [--k 24] [--out palette.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Files in the icons dir that are not species art
EXCLUDE = {"icon-192.png", "icon-512.png"}


def sample_pixels(png: Path, per_image: int, rng: np.random.Generator) -> np.ndarray:
    im = Image.open(png).convert("RGBA")
    # downsample large canvases first; palette doesn't need full res
    if max(im.size) > 512:
        im.thumbnail((512, 512), Image.LANCZOS)
    arr = np.asarray(im)
    opaque = arr[arr[..., 3] > 200][:, :3]  # ignore soft edges
    if len(opaque) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    idx = rng.choice(len(opaque), size=min(per_image, len(opaque)), replace=False)
    return opaque[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icons", default="frontend/public/icons")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--per-image", type=int, default=4000)
    ap.add_argument("--out", default="art-pipeline/palette.json")
    args = ap.parse_args()

    rng = np.random.default_rng(seed=7)
    pngs = [
        p for p in sorted(Path(args.icons).glob("*.png"))
        if p.name not in EXCLUDE and "_old" not in p.name
    ]
    print(f"sampling {len(pngs)} illustrations...")
    samples = np.vstack([sample_pixels(p, args.per_image, rng) for p in pngs])
    print(f"{len(samples):,} opaque pixels sampled")

    km = KMeans(n_clusters=args.k, n_init=4, random_state=7).fit(samples.astype(np.float32))
    counts = np.bincount(km.labels_, minlength=args.k)
    order = np.argsort(-counts)

    palette = []
    for i in order:
        r, g, b = (int(round(c)) for c in km.cluster_centers_[i])
        palette.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [r, g, b],
            "share": round(float(counts[i]) / len(samples), 4),
        })

    Path(args.out).write_text(json.dumps(palette, indent=2))
    print(f"wrote {args.out}")
    for entry in palette:
        print(f"  {entry['hex']}  {entry['share']:>6.1%}")


if __name__ == "__main__":
    main()
