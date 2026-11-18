"""Isolated test for mammal fur texture & matte flat color steps (Marten / Macaque style vs Watercolory Gouache swirls).

Tests on American Pika, California Ground Squirrel, Muskrat, and Mule Deer:
1. Col 1: Current v5 baseline ('gouache texture marks' in positive, 'pen hatching' in negative, CFG=7.0)
2. Col 2: v5 + Ink-Line / Flat Matte Prompt (removes 'gouache', removes 'pen hatching' from neg, adds 'watercolor, gouache, painterly brush swirls, glossy' to neg, CFG=5.5)
3. Col 3: v5 (UNet-only via --network_train_unet_only) + Ink-Line / Flat Matte Prompt (CFG=5.5, s=0.78)
4. Col 4: v4-step1250 (trained with zero style words in captions) + Ink-Line / Flat Matte Prompt (CFG=5.5, s=0.76)
"""

import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
TEST_DIR = Path(__file__).parent / "batch_run_mammal_ink_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity/brain/cf192390-434f-49ae-a401-59243749f021")

MAMMALS = [
    (4, "mule_deer", "Mule Deer"),
    (5, "muskrat", "Muskrat"),
    (6, "california_ground_squirrel", "California Ground Squirrel"),
    (7, "american_pika", "American Pika"),
]

# Notice: we remove 'cross-hatching, pen hatching' from NEG and add 'watercolor, gouache, painterly brush swirls, soft blended fur, airbrushed, glossy, shiny, specular sheen'
NEG_INK_MATTE = (
    "photo, photorealistic, 3d render, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, soft blended brushwork, oil painting, blurry fur, "
    "heavy black sticker outline, "
    "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, extra limbs, tiling, text, watermark, border, cropped body, cut off"
)

STYLE_INK_MATTE = (
    "clean stylized 2d wildlife illustration, flat matte color steps, "
    "fine dark ink fur linework and crisp stylized ink marks like pen and flat color, "
    "clean thin dark charcoal-umber outline, muted natural matte palette, "
    "isolated on a plain solid white background"
)


def clean_and_center(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    im = Image.open(src).convert("RGBA")
    cut = remove(im)
    box = subject_bbox(cut)
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    canvas.convert("RGB").save(dst)


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = TEST_DIR / "inputs"
    in_dir.mkdir(exist_ok=True)

    vlm_cache = json.loads((V5_EVAL / "vlm_11_cache_v2.json").read_text(encoding="utf-8"))

    prompts_v5 = []
    prompts_v4 = []
    for i, (orig_idx, slug, common) in enumerate(MAMMALS):
        src_inp = V5_EVAL / "inputs" / f"{orig_idx:02d}_{slug}.png"
        dst_inp = in_dir / f"{i:02d}_{slug}.png"
        Image.open(src_inp).convert("RGB").save(dst_inp)

        res = vlm_cache[slug]
        article = "an" if common[0].lower() in "aeiou" else "a"
        missing = res.get("missing_parts_to_fill", "").strip()
        colors = res.get("visible_color_and_markings", "").strip()
        # Replace soft words like 'dense fur' with 'matte coat with ink fur marks'
        colors_clean = colors.replace("dense fur", "flat matte coat").replace("medium brown fur", "medium brown coat")

        if missing:
            p5 = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal {missing}, {colors_clean}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
            p4 = f"{article} {common.lower()}, full body wildlife illustration, single animal {missing}, {colors_clean}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
        else:
            p5 = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal, {colors_clean}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
            p4 = f"{article} {common.lower()}, full body wildlife illustration, single animal, {colors_clean}, {STYLE_INK_MATTE} --n {NEG_INK_MATTE} --d 42"
        prompts_v5.append(p5)
        prompts_v4.append(p4)

    pfile_v5 = TEST_DIR / "prompts_v5.txt"
    pfile_v4 = TEST_DIR / "prompts_v4.txt"
    pfile_v5.write_text("\n".join(prompts_v5), encoding="utf-8")
    pfile_v4.write_text("\n".join(prompts_v4), encoding="utf-8")

    runs = [
        ("col2_v5_ink_cfg55", LORA / "output" / "animl_style_v5-step00001000.safetensors", pfile_v5, "0.75", "5.5", False),
        ("col3_v5_unetonly_s078", LORA / "output" / "animl_style_v5-step00001250.safetensors", pfile_v5, "0.78", "5.5", True),
        ("col4_v4_ink_s076", LORA / "output" / "animl_style_v4-step00001250.safetensors", pfile_v4, "0.76", "5.5", True),
    ]

    for tag, ckpt, pf, s, cfg, unet_only in runs:
        out_sub = TEST_DIR / f"out_{tag}"
        clean_sub = TEST_DIR / f"clean_{tag}"
        out_sub.mkdir(exist_ok=True)
        clean_sub.mkdir(exist_ok=True)
        if len(list(clean_sub.glob("*.png"))) < len(MAMMALS):
            cmd = [
                str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
                "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
                "--outdir", str(out_sub),
                "--network_module", "networks.lora",
                "--network_weights", str(ckpt),
                "--network_mul", "1.0",
                "--max_embeddings_multiples", "3",
                "--from_file", str(pf),
                "--image_path", str(in_dir),
                "--use_original_file_name",
                "--strength", s,
                "--W", "1024", "--H", "1024",
                "--steps", "28", "--sampler", "euler_a",
                "--scale", cfg, "--bf16", "--sdpa",
                "--batch_size", "1", "--images_per_prompt", "1",
            ]
            subprocess.run(cmd, check=True)
            for p in out_sub.glob("*.png"):
                clean_and_center(p, clean_sub / p.name)

    # Render comparison sheet alongside American Marten & Rhesus Macaque reference crops at top
    cw, ch = 360, 360
    cols = [
        "1. Prev v5 ('gouache', CFG=7.0)",
        "2. v5 + Ink/Matte Prompt (CFG=5.5)",
        "3. v5 UNet-Only + Ink/Matte (s=0.78)",
        "4. v4 UNet-Only + Ink/Matte (s=0.76)",
    ]
    W = cw * 4
    H = 48 + len(MAMMALS) * (ch + 36)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arialbd.ttf", 18)
        font_sm = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    for ci, t in enumerate(cols):
        d.text((ci * cw + 10, 12), t, fill=(20, 20, 20), font=font)

    for ri, (orig_idx, slug, common) in enumerate(MAMMALS):
        y_top = 48 + ri * (ch + 36)
        d.rectangle([0, y_top, W, y_top + 36], fill=(232, 232, 236))
        d.text((12, y_top + 8), f"{common} ({slug})", fill=(10, 10, 80), font=font_sm)

        files = [
            V5_EVAL / "clean_step1000_s0.75" / f"{orig_idx:02d}_{slug}.png",
            TEST_DIR / "clean_col2_v5_ink_cfg55" / f"{ri:02d}_{slug}.png",
            TEST_DIR / "clean_col3_v5_unetonly_s078" / f"{ri:02d}_{slug}.png",
            TEST_DIR / "clean_col4_v4_ink_s076" / f"{ri:02d}_{slug}.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 36))
            d.rectangle([ci * cw, y_top + 36, (ci + 1) * cw - 1, y_top + 36 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "mammal_ink_vs_gouache_compare.jpg", quality=92)
    print("Saved mammal_ink_vs_gouache_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
