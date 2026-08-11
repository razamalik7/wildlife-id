"""Build the SDXL style-LoRA training dataset from the art masters.

Per the renovation plan: training data is illustrations + content-only
captions. No style words in captions (style must bake in as default
behavior, not an optional attribute). Reference photos are NOT training
data.

Each master is alpha-cropped, scaled to 76% fill, composited onto white
(SDXL has no alpha channel), saved at 1024px with a caption file
"a(n) {species name}".

Usage: python art-pipeline/lora/build_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "art-pipeline"))
from normalize import subject_bbox  # noqa: E402

MASTERS = REPO / "art-pipeline" / "masters"
OUT = Path(__file__).parent / "dataset" / "img"
SIZE = 1024
FILL = 0.76

# v3: exclude the punchy ink-sticker/graphic outliers so the LoRA learns the
# soft matte painterly core (Claude visual pass, 2026-08-11; user can veto).
# Vignettes (water pools, branches, rocks) remain represented via the core.
EXCLUDE: set[str] = {
    "american_flamingo", "american_red_squirrel", "bald_eagle", "blue_jay",
    "bobcat", "eastern_newt", "mallard", "monk_parakeet", "moose",
    "northern_cardinal", "red_fox",
}


def caption_for(slug: str) -> str:
    name = slug.replace("_", " ")
    article = "an" if name[0] in "aeiou" else "a"
    return f"{article} {name}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(MASTERS.glob("*.png")):
        if png.stem in EXCLUDE:
            print(f"skip {png.stem} (excluded)")
            continue
        im = Image.open(png).convert("RGBA")
        box = subject_bbox(im)
        if box is None:
            continue
        subject = im.crop(box)
        target = int(SIZE * FILL)
        scale = target / max(subject.size)
        nw, nh = (max(1, round(d * scale)) for d in subject.size)
        subject = subject.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
        canvas.paste(subject, ((SIZE - nw) // 2, (SIZE - nh) // 2), subject)
        canvas.convert("RGB").save(OUT / f"{png.stem}.png")
        (OUT / f"{png.stem}.txt").write_text(caption_for(png.stem))
        n += 1
    print(f"dataset: {n} image/caption pairs in {OUT}")


if __name__ == "__main__":
    main()
