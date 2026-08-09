"""Post-normalization for accepted illustrations.

Crops each RGBA image to its alpha bounding box and rescales so the subject's
long side fills a fixed fraction of a square canvas, centered. This removes
scale inconsistency between pieces programmatically — scale is not an art
problem. True body size belongs in species data, not in relative art scale.

Usage:
    python art-pipeline/normalize.py --in frontend/public/icons/Unscaled --out out/
    python art-pipeline/normalize.py --in some.png --out out/ --canvas 2048 --fill 0.76

Never overwrites input files; refuses if --out resolves into --in.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ALPHA_THRESHOLD = 10  # ignore near-transparent halo pixels when boxing


def subject_bbox(im: Image.Image) -> tuple[int, int, int, int] | None:
    a = np.asarray(im)[..., 3]
    ys, xs = np.where(a > ALPHA_THRESHOLD)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalize(png: Path, out_dir: Path, canvas: int, fill: float) -> str:
    im = Image.open(png).convert("RGBA")
    box = subject_bbox(im)
    if box is None:
        return f"SKIP {png.name}: fully transparent"
    l, t, r, b = box
    w, h = r - l, b - t
    pre_fill = max(w, h) / max(im.size)

    subject = im.crop(box)
    target_long = int(round(canvas * fill))
    scale = target_long / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    subject = subject.resize((nw, nh), resample)

    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.paste(subject, ((canvas - nw) // 2, (canvas - nh) // 2))
    out.save(out_dir / png.name)
    return f"OK   {png.name}: fill {pre_fill:.0%} -> {fill:.0%}, upscale x{scale:.2f}" if scale > 1 \
        else f"OK   {png.name}: fill {pre_fill:.0%} -> {fill:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="input png or directory")
    ap.add_argument("--out", dest="out", required=True, help="output directory")
    ap.add_argument("--canvas", type=int, default=2048)
    ap.add_argument("--fill", type=float, default=0.76,
                    help="fraction of canvas the subject's long side fills")
    args = ap.parse_args()

    src, out_dir = Path(args.src), Path(args.out)
    pngs = sorted(src.glob("*.png")) if src.is_dir() else [src]
    if not pngs:
        sys.exit(f"no .png files under {src}")
    if out_dir.resolve() == (src if src.is_dir() else src.parent).resolve():
        sys.exit("refusing to write into the input directory; pick a different --out")
    out_dir.mkdir(parents=True, exist_ok=True)

    for png in pngs:
        print(normalize(png, out_dir, args.canvas, args.fill))


if __name__ == "__main__":
    main()
