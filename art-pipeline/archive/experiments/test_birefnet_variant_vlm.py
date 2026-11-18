"""Test BiRefNet cutouts + refined gemma3:27b variant-matching prompt on the 6 unseen species.
Verifies:
1. Pond Slider has ZERO log branch, brownish-olive adult shell, dark green head/limbs with yellow lines, and red stripe behind eye.
2. Green-winged Teal has its complete dark bill intact + webbed feet.
3. Mule Deer has straight standing front legs + cloven hooves.
"""

import base64
import json
import re
import shutil
import subprocess
import time
from io import BytesIO
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from u2net_matte import remove
from normalize import subject_bbox

OLLAMA = "http://localhost:11434"
LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
BIREF_TEST = WORK / "birefnet_vlm_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity/brain/cf192390-434f-49ae-a401-59243749f021")

SPECIES = [
    ("mule_deer", "Mule Deer", "Mule deer", "grass_hooves"),
    ("pond_slider", "Pond Slider", "Red-eared slider", "swimming"),
    ("green-winged_teal", "Green-winged Teal", "Green-winged teal", "floating"),
    ("american_pika", "American Pika", "American pika", "rock_perch"),
    ("california_ground_squirrel", "California Ground Squirrel", "California ground squirrel", "standing"),
    ("common_watersnake", "Common Watersnake", "Common watersnake", "coiled"),
]


def build_birefnet_cutout(slug: str) -> Path:
    photo_p = WORK / f"{slug}_photo.png"
    cut_p = BIREF_TEST / f"{slug}_biref_cutout.png"
    if not cut_p.exists():
        photo = Image.open(photo_p).convert("RGBA")
        cut = remove(photo)
        box = subject_bbox(cut)
        subj = cut.crop(box)
        scale = int(1024 * 0.76) / max(subj.size)
        nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
        subj = subj.resize((nw, nh), Image.LANCZOS)
        canv = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canv.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
        canv.convert("RGB").save(cut_p)
    return cut_p


def make_soft_edge_cutout(cut_p: Path, dst_p: Path, mode: str) -> None:
    im = Image.open(cut_p).convert("RGB")
    if mode in ("grass_hooves", "rock_perch", "swimming", "floating"):
        arr = np.array(im)
        mask = (arr < 248).any(axis=2)
        ys, xs = np.where(mask)
        if len(ys) > 0:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            crop = im.crop((x0, y0, x1 + 1, y1 + 1))
            cw, ch = crop.size
            scale = int(1024 * 0.68) / max(cw, ch)
            nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
            crop_resized = crop.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGB", (1024, 1024), (255, 255, 255))
            px = (1024 - nw) // 2
            py = max(40, (1024 - nh) // 2 - 45)
            canvas.paste(crop_resized, (px, py))
            if mode in ("grass_hooves", "swimming", "rock_perch"):
                strip_h = min(24, nh // 6)
                bottom_strip = crop_resized.crop((0, nh - strip_h, nw, nh))
                ext_h = 48 if mode != "grass_hooves" else 64
                bottom_ext = bottom_strip.resize((nw, ext_h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(8))
                white_bg = Image.new("RGB", (nw, ext_h), (255, 255, 255))
                seeded = Image.blend(white_bg, bottom_ext, 0.42)
                canvas.paste(seeded, (px, py + nh - 6))
            canvas.save(dst_p)
            return
    shutil.copyfile(cut_p, dst_p)


def fetch_full_wiki_description(title: str) -> str:
    s = requests.Session()
    s.headers.update({"User-Agent": "AnimlVisionBot/1.0"})
    r = s.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1, "format": "json"},
        timeout=20,
    ).json()
    page = list(r["query"]["pages"].values())[0].get("extract", "")
    lead = page.split("==")[0].strip()
    sections = re.findall(
        r"==+\s*(?:Description|Appearance|Physical description|Coloration|Plumage|Characteristics).*?==+(.+?)(?===+\s*\w|$)",
        page,
        re.S | re.I,
    )
    combined = lead + "\n\n" + "\n".join(sections)
    return re.sub(r"\s+", " ", combined)[:3000]


PROMPT_TEMPLATE = """You are a wildlife illustrator inspecting a cutout image of a {common_name} alongside its Wikipedia Description.

Wikipedia Description for {common_name}:
\"\"\"{wiki_text}\"\"\"

Your job is to return ONLY two short fields in JSON:
1. `missing_parts_to_fill`: Look at the bottom/edges of the animal cutout.
   - If the animal's lower legs, hooves, webbed feet, paws, or tail are cut off or hidden:
     * For standing hoofed mammals (deer, elk, goats, etc.) whose lower legs are cut off, write: `"standing on all four slender legs with dark cloven hooves visible"`
     * For floating/standing ducks or waterbirds whose feet are cut off, write: `"with webbed feet visible underneath"`
     * For small crouching mammals or amphibians whose bottom paws are cut off, write: `"with rounded hind haunch and all four paws visible"`
     * For swimming turtles whose feet are tucked/hidden, write: `"with webbed clawed swimming feet visible"`
   - If the animal's body and limbs/coils are already complete (like a coiled snake or standing squirrel with paws visible), return `""`.
2. `visible_color_and_markings`:
   - CRITICAL VARIANT RULE: When there are age, sex, or seasonal variants listed in the Wikipedia description (e.g., leaf-green hatchling vs brownish-olive adult turtle shell, or breeding male vs female duck), look closely at the attached image to determine WHICH variant applies to this individual, and use ONLY that variant's colors together with the species' signature visible markings from Wikipedia!
   - Write a 15-22 word comma-separated list of `[color/pattern] [body part]` phrases for ONLY the parts of the animal visible from this angle (e.g., for an adult Pond Slider: `'brownish-olive oval carapace, dark green head and limbs with fine yellow lines, distinct red stripe behind eye'`).
   - DO NOT mention hidden underside parts (no 'plastron', 'ventral scales', 'underwing coverts').
   - DO NOT describe pose or camera angle.
   - DO NOT mention environment, water, grass, rocks, or logs.
   - DO NOT use the word 'pinkish'.

Return STRICT JSON only:
{{
  "missing_parts_to_fill": "short phrase for cut-off anatomy, or empty string",
  "visible_color_and_markings": "15-22 word comma-separated list of visible `[color/pattern] [body part]` phrases matching the variant in the image"
}}"""


def call_gemma27b(cut_p: Path, common_name: str, wiki_text: str) -> dict:
    im = Image.open(cut_p).convert("RGB")
    im.thumbnail((768, 768), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = PROMPT_TEMPLATE.format(common_name=common_name, wiki_text=wiki_text)
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": "gemma3:27b",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        },
        timeout=180,
    )
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


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
    BIREF_TEST.mkdir(parents=True, exist_ok=True)
    in_dir = BIREF_TEST / "inputs"
    out_dir = BIREF_TEST / "outputs"
    clean_dir = BIREF_TEST / "clean"
    for d in (in_dir, out_dir, clean_dir):
        d.mkdir(exist_ok=True)

    NEG_BASE = (
        "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
        "heavy black outline, sticker, cross-hatching, pen hatching, "
        "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
        "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
        "multiple animals, duplicate, two heads, extra limbs, tiling, text, watermark, border, cropped body, cut off, "
        "box turtle, speckled background, confetti"
    )
    STYLE_CLAUSE = (
        "rich natural colors, crisp distinct painted color steps, flat stylized "
        "gouache texture marks, clean thin dark charcoal-umber outline, "
        "isolated on a plain solid white background"
    )

    prompts = []
    vlm_out = {}
    for i, (slug, common, wiki_title, mode) in enumerate(SPECIES):
        biref_cut = build_birefnet_cutout(slug)
        inp_p = in_dir / f"{i:02d}_{slug}.png"
        make_soft_edge_cutout(biref_cut, inp_p, mode)

        wiki_txt = fetch_full_wiki_description(wiki_title)
        res = call_gemma27b(biref_cut, common, wiki_txt)
        vlm_out[slug] = res
        print(f"\n[{slug}]:\n  Fill: {res.get('missing_parts_to_fill')!r}\n  Colors: {res.get('visible_color_and_markings')!r}", flush=True)

        article = "an" if common[0].lower() in "aeiou" else "a"
        missing = res.get("missing_parts_to_fill", "").strip()
        colors = res.get("visible_color_and_markings", "").strip()

        if missing:
            p = f"{article} {common.lower()}, full body wildlife illustration, single animal {missing}, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        else:
            p = f"{article} {common.lower()}, full body wildlife illustration, single animal, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        prompts.append(p)

    (BIREF_TEST / "vlm_outputs.json").write_text(json.dumps(vlm_out, indent=2), encoding="utf-8")
    pfile = BIREF_TEST / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    requests.post(f"{OLLAMA}/api/generate", json={"model": "gemma3:27b", "keep_alive": 0}, timeout=15)
    time.sleep(2)

    subprocess.run([
        str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
        "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        "--outdir", str(out_dir),
        "--network_module", "networks.lora",
        "--network_weights", str(LORA / "output" / "animl_style_v4-step00001250.safetensors"),
        "--network_mul", "1.0",
        "--max_embeddings_multiples", "3",
        "--from_file", str(pfile),
        "--image_path", str(in_dir),
        "--use_original_file_name",
        "--strength", "0.76",
        "--W", "1024", "--H", "1024",
        "--steps", "28", "--sampler", "euler_a",
        "--scale", "7.0", "--bf16", "--sdpa",
        "--batch_size", "1", "--images_per_prompt", "1",
    ], check=True)

    for p in out_dir.glob("*.png"):
        clean_and_center(p, clean_dir / p.name)

    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
        font_sm = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    cols = [
        "1. New BiRefNet Cutout (1024px)",
        "2. Old u2net + No VLM",
        "3. Prev u2net + Fill-First VLM",
        "4. BiRefNet + Variant-Aware VLM",
    ]
    cw, ch = 360, 360
    W = len(cols) * cw
    H = 48 + len(SPECIES) * (ch + 44)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 12, 12), t, fill=(20, 20, 20), font=font)

    old_stems = ["04_mule_deer", "08_pond_slider", "03_green-winged_teal", "07_american_pika", "06_california_ground_squirrel", "09_common_watersnake"]
    for ri, (slug, common, _, _) in enumerate(SPECIES):
        y_top = 48 + ri * (ch + 44)
        d.rectangle([0, y_top, W, y_top + 44], fill=(232, 232, 236))
        m_txt = vlm_out[slug].get("missing_parts_to_fill", "") or "(none)"
        c_txt = vlm_out[slug].get("visible_color_and_markings", "")
        d.text((12, y_top + 4), f"{common} | Fill: {m_txt}", fill=(10, 10, 80), font=font_sm)
        d.text((12, y_top + 23), f"Variant Colors: {c_txt[:135]}", fill=(30, 30, 30), font=font_sm)

        files = [
            in_dir / f"{ri:02d}_{slug}.png",
            WORK / "clean_str_0.76" / f"{old_stems[ri]}.png",
            WORK / "zero_pose_vlm_test" / "clean" / f"{ri:02d}_{slug}.png",
            clean_dir / f"{ri:02d}_{slug}.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 44))
            d.rectangle([ci * cw, y_top + 44, (ci + 1) * cw - 1, y_top + 44 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "birefnet_variant_vlm_compare.jpg", quality=92)
    print("Saved birefnet_variant_vlm_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
