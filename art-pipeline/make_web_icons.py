"""Generate web-serving derivatives of the species illustrations.

The masters in frontend/public/icons are 1024-2048px PNGs up to 3.8 MB —
far heavier than any display size in the app (grid ~100 px, detail hero
192 css px). Emit 1024px WebP files, which keep a large quality margin
while cutting ~95% of the transfer.

Usage:
    python art-pipeline/make_web_icons.py [--icons art-pipeline/masters]
                                          [--out frontend/public/icons/web]
"""

import argparse
from pathlib import Path

from PIL import Image

EXCLUDE_PREFIXES = ("icon-",)
EXCLUDE_MARKERS = ("_old", "_alt")
SIZE = 1024
QUALITY = 90


def is_species_art(p: Path) -> bool:
    return (
        not p.name.startswith(EXCLUDE_PREFIXES)
        and not any(m in p.stem for m in EXCLUDE_MARKERS)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icons", default="art-pipeline/masters")
    ap.add_argument("--out", default="frontend/public/icons/web")
    args = ap.parse_args()

    icons, out = Path(args.icons), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total_in = total_out = count = 0
    for png in sorted(icons.glob("*.png")):
        if not is_species_art(png):
            continue
        im = Image.open(png).convert("RGBA")
        if max(im.size) > SIZE:
            im.thumbnail((SIZE, SIZE), Image.LANCZOS)
        dest = out / (png.stem + ".webp")
        im.save(dest, "WEBP", quality=QUALITY, method=6)
        total_in += png.stat().st_size
        total_out += dest.stat().st_size
        count += 1

    print(f"{count} icons: {total_in/1e6:.0f} MB PNG -> {total_out/1e6:.1f} MB WebP")


if __name__ == "__main__":
    main()
