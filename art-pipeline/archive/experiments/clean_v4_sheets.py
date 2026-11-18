"""Normalize all 33 generated v4 unseen test images through u2net_matte + 76% fill
and build clean side-by-side contact sheets.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

WORK = Path(__file__).parent / "batch_run_v4"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

UNSEEN_SPECIES = [
    ("northern_shoveler", "Northern Shoveler", "floating"),
    ("northern_pintail", "Northern Pintail", "floating"),
    ("american_wigeon", "American Wigeon", "floating"),
    ("green-winged_teal", "Green-winged Teal", "floating"),
    ("mule_deer", "Mule Deer", "grass_hooves"),
    ("muskrat", "Muskrat", "swimming"),
    ("california_ground_squirrel", "California Ground Squirrel", "standing"),
    ("american_pika", "American Pika", "rock_perch"),
    ("pond_slider", "Pond Slider", "swimming"),
    ("common_watersnake", "Common Watersnake", "coiled"),
    ("gopher_tortoise", "Gopher Tortoise", "walking"),
]


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
    strengths = [0.68, 0.76, 0.84]
    for s in strengths:
        outdir = WORK / f"out_str_{s}"
        cleandir = WORK / f"clean_str_{s}"
        cleandir.mkdir(exist_ok=True)
        for idx, (slug, _, _) in enumerate(UNSEEN_SPECIES):
            src = outdir / f"{idx:02d}_{slug}.png"
            dst = cleandir / f"{idx:02d}_{slug}.png"
            if src.exists():
                clean_and_center(src, dst)
        print(f"Cleaned strength {s}", flush=True)

    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    groups = [
        ("v4_unseen_land_and_semi_aquatic.jpg", [x for x in UNSEEN_SPECIES if x[2] != "floating"]),
        ("v4_unseen_waterfowl.jpg", [x for x in UNSEEN_SPECIES if x[2] == "floating"]),
    ]

    in_dir = WORK / "inputs"
    for fname, items in groups:
        cols = ["Reference Photo", "Cutout Input", "v4 LoRA @ 0.68", "v4 LoRA @ 0.76", "v4 LoRA @ 0.84"]
        cell_w, cell_h = 340, 340
        header_h = 48
        row_label_h = 32
        W = len(cols) * cell_w
        H = header_h + len(items) * (cell_h + row_label_h)
        sheet_im = Image.new("RGB", (W, H), (248, 248, 248))
        draw = ImageDraw.Draw(sheet_im)

        for ci, col_title in enumerate(cols):
            draw.text((ci * cell_w + 14, 12), col_title, fill=(20, 20, 20), font=font)

        for ri, (slug, common, mode) in enumerate(items):
            idx = [x[0] for x in UNSEEN_SPECIES].index(slug)
            y_top = header_h + ri * (cell_h + row_label_h)
            draw.rectangle([0, y_top, W, y_top + row_label_h], fill=(232, 232, 236))
            draw.text((14, y_top + 5), f"{common} ({slug}) — mode: {mode}", fill=(10, 10, 10), font=font_sm)

            row_files = [
                WORK / f"{slug}_photo.png",
                in_dir / f"{idx:02d}_{slug}.png",
                WORK / "clean_str_0.68" / f"{idx:02d}_{slug}.png",
                WORK / "clean_str_0.76" / f"{idx:02d}_{slug}.png",
                WORK / "clean_str_0.84" / f"{idx:02d}_{slug}.png",
            ]
            for ci, fp in enumerate(row_files):
                cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
                if fp.exists():
                    im = Image.open(fp).convert("RGB")
                    im.thumbnail((cell_w - 8, cell_h - 8), Image.LANCZOS)
                    cell.paste(im, ((cell_w - im.width) // 2, (cell_h - im.height) // 2))
                sheet_im.paste(cell, (ci * cell_w, y_top + row_label_h))
                draw.rectangle(
                    [ci * cell_w, y_top + row_label_h, (ci + 1) * cell_w - 1, y_top + row_label_h + cell_h - 1],
                    outline=(210, 210, 210),
                )

        sheet_im.save(WORK / fname, quality=92)
        sheet_im.save(ARTIFACT_DIR / fname, quality=92)
        print(f"Saved cleaned {fname}", flush=True)


if __name__ == "__main__":
    main()
