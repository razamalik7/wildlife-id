"""Turn an accepted generated illustration into app-ready assets.

Input: a generated PNG on flat white (from generate_batch.py output dirs).
Steps: rembg matte (white -> alpha) -> alpha-bbox crop -> 76% fill centered
on a transparent 2048 canvas (master) -> 1024px WebP for the app.

The continuous dark outline in the house style gives rembg a clean edge to
cut along; spot-check masters against a dark background anyway.

Usage:
    python art-pipeline/finalize_art.py <generated.png> --slug wood_duck
        [--masters art-pipeline/masters] [--web frontend/public/icons/web]
        [--overwrite]
"""

import argparse
import sys
from pathlib import Path

from PIL import Image
from rembg import remove

sys.path.insert(0, str(Path(__file__).parent))
from normalize import subject_bbox  # noqa: E402

CANVAS = 2048
FILL = 0.76
WEB = 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("generated", help="accepted generated PNG (white background)")
    ap.add_argument("--slug", required=True, help="species slug, e.g. wood_duck")
    ap.add_argument("--masters", default="art-pipeline/masters")
    ap.add_argument("--web", default="frontend/public/icons/web")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    master_p = Path(args.masters) / f"{args.slug}.png"
    web_p = Path(args.web) / f"{args.slug}.webp"
    for p in (master_p, web_p):
        if p.exists() and not args.overwrite:
            sys.exit(f"{p} exists - pass --overwrite to replace existing art")

    im = Image.open(args.generated).convert("RGBA")
    cut = remove(im)
    box = subject_bbox(cut)
    if box is None:
        sys.exit("matte produced no subject")
    subject = cut.crop(box)

    scale = int(CANVAS * FILL) / max(subject.size)
    nw, nh = (max(1, round(d * scale)) for d in subject.size)
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    subject = subject.resize((nw, nh), resample)
    master = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    master.paste(subject, ((CANVAS - nw) // 2, (CANVAS - nh) // 2), subject)
    master.save(master_p)

    web = master.copy()
    web.thumbnail((WEB, WEB), Image.LANCZOS)
    web_p.parent.mkdir(parents=True, exist_ok=True)
    web.save(web_p, "WEBP", quality=90, method=6)
    print(f"master: {master_p}\nweb:    {web_p}")


if __name__ == "__main__":
    main()
