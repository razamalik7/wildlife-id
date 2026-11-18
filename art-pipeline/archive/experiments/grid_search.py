"""Overnight recipe grid search for the art factory.

Sweeps LoRA version x denoise strength x LoRA weight with the matte prompt
language on test species, writing one dir per config. Review strips are
built per species afterward.

Usage: python art-pipeline/grid_search.py  (uses lora venv's python for gen)
"""

import itertools
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
LORA = ROOT / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
OUT = ROOT / "batch_run" / "grid"

SPECIES = {
    "muskrat": "a muskrat, full body wildlife illustration, muted matte colors, soft painterly brushwork, flat gouache shading, long thin hairless tail visible, resting on a small stylized pool of blue water with simple ripple lines, white background",
    "american_wigeon": "an american wigeon duck, full body wildlife illustration, muted matte colors, soft painterly brushwork, flat gouache shading, resting on a small stylized pool of blue water with simple ripple lines, white background",
    "great_egret": "a great egret, full body wildlife illustration, muted matte colors, soft painterly brushwork, flat gouache shading, standing with legs in a shallow stylized pool of blue water, white background",
}
NEG = ("sticker, heavy black outline, glossy, shiny, oversaturated, 3d render, "
       "airbrushed, photo, photorealistic, multiple animals, speckles, confetti, "
       "dots, text, watermark, cropped body")

CONFIGS = [
    {"lora": v, "strength": s, "mul": m}
    for v, s, m in itertools.product(
        ["animl_style_v1", "animl_style_v2"], [0.5, 0.6], [1.0, 1.15])
]


def main() -> None:
    for slug, prompt in SPECIES.items():
        cut = ROOT / "batch_run" / f"{slug}_cutout.png"
        for c in CONFIGS:
            tag = f"{c['lora'][-2:]}_s{c['strength']}_m{c['mul']}"
            outdir = OUT / slug / tag
            if list(outdir.glob("*.png")):
                continue
            subprocess.run([
                str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
                "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
                "--outdir", str(outdir),
                "--network_module", "networks.lora",
                "--network_weights", str(LORA / "output" / f"{c['lora']}.safetensors"),
                "--network_mul", str(c["mul"]),
                "--prompt", f"{prompt} --n {NEG}",
                "--image_path", str(cut),
                "--strength", str(c["strength"]),
                "--W", "1024", "--H", "1024", "--steps", "32",
                "--sampler", "euler_a", "--scale", "7", "--seed", "123",
                "--bf16", "--sdpa", "--batch_size", "1", "--images_per_prompt", "1",
            ], check=True, capture_output=True)
            print(f"done {slug} {tag}", flush=True)

    # review strips: one per species, configs in fixed order
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    T = 340
    for slug in SPECIES:
        tags = [f"{c['lora'][-2:]}_s{c['strength']}_m{c['mul']}" for c in CONFIGS]
        strip = Image.new("RGB", (T * len(tags), T + 30), "white")
        d = ImageDraw.Draw(strip)
        for i, tag in enumerate(tags):
            files = sorted((OUT / slug / tag).glob("*.png"))
            if files:
                im = Image.open(files[0])
                im.thumbnail((T, T))
                strip.paste(im, (i * T, 30))
            d.text((i * T + 6, 4), tag, fill="black", font=font)
        strip.save(OUT / f"{slug}_grid.png")
    print("grids built")


if __name__ == "__main__":
    main()
