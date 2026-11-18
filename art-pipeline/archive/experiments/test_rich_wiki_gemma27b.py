"""Test gemma3:27b with full Wikipedia Description section + Cutout image,
generating a rich 20-25 word field-guide description (with ZERO pose text!)
and running SDXL img2img at Step 1250 @ 0.76 so we can see the Pond Slider's
brownish shell + dark green head/limbs with fine yellow lines + red ear stripe behind the eye.
"""

import base64
import json
import re
import shutil
import subprocess
import time
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

OLLAMA = "http://localhost:11434"
LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
RICH_DIR = WORK / "rich_wiki_vlm_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SPECIES = [
    ("04_mule_deer", "mule_deer", "Mule Deer", "Mule deer",
     "standing with all four legs and clean hooves visible"),
    ("08_pond_slider", "pond_slider", "Pond Slider", "Red-eared slider",
     "in swimming pose with all limbs visible"),
    ("03_green-winged_teal", "green-winged_teal", "Green-winged Teal", "Green-winged teal",
     "with webbed feet visible"),
    ("07_american_pika", "american_pika", "American Pika", "American pika",
     "with complete rounded body and all four paws visible"),
    ("06_california_ground_squirrel", "california_ground_squirrel", "California Ground Squirrel", "California ground squirrel",
     "with complete body and paws visible"),
    ("09_common_watersnake", "common_watersnake", "Common Watersnake", "Common watersnake",
     "with complete coiled body visible"),
]


def fetch_full_wiki_description(title: str) -> str:
    s = requests.Session()
    s.headers.update({"User-Agent": "AnimlVisionBot/1.0"})
    r = s.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1, "format": "json"},
        timeout=20,
    ).json()
    page = list(r["query"]["pages"].values())[0].get("extract", "")
    # Grab lead + full Description/Appearance/Coloration sections (up to 3500 chars)
    lead = page.split("==")[0].strip()
    sections = re.findall(
        r"==+\s*(?:Description|Appearance|Physical description|Coloration|Plumage|Characteristics).*?==+(.+?)(?===+\s*\w|$)",
        page,
        re.S | re.I,
    )
    combined = lead + "\n\n" + "\n".join(sections)
    return re.sub(r"\s+", " ", combined)[:3000]


PROMPT_TEMPLATE = """You are a field-guide ornithologist and herpetologist writing a precise visual coloration description for a {common_name} illustration.

Here is the Wikipedia Description text for {common_name}:
\"\"\"{wiki_text}\"\"\"

Look at the attached cutout of the individual {common_name} AND the Wikipedia description above.
Synthesize a rich, accurate 18-28 word comma-separated list of `[color/pattern] [body part]` phrases that combines:
1. The overall coloration visible on this individual (e.g. if the shell is brownish-olive on this adult, say brownish-olive carapace, not leaf-green hatchling).
2. The signature species markings from Wikipedia (e.g., for Pond Slider: 'brownish-olive carapace with faint yellow lines, dark green head and limbs with fine yellow stripes, distinct red stripe behind the eye'; for Green-winged Teal: 'chestnut head with iridescent green eye-patch, vertical white breast stripe, speckled buff breast, finely barred grey flanks, pale yellow undertail patch').

CRITICAL RULES:
- DO NOT describe pose, camera angle, or view (no 'standing', 'lying', 'side view').
- DO NOT mention any environment, water, grass, rocks, or logs.
- DO NOT use the word 'pinkish' (use 'warm buff' or 'tan-buff' instead).
- Attach every color/marking directly to its body part (e.g. 'red stripe behind the eye', NOT 'red eye').

Return STRICT JSON only:
{{
  "species_color_and_markings": "18-28 word comma-separated list of precise `[color/pattern] [body part]` phrases"
}}"""


def call_gemma27b_rich(cut_p: Path, common_name: str, wiki_text: str) -> str:
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
    data = json.loads(r.json()["message"]["content"])
    return data.get("species_color_and_markings", "")


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
    RICH_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = RICH_DIR / "inputs"
    out_dir = RICH_DIR / "outputs"
    clean_dir = RICH_DIR / "clean"
    for d in (in_dir, out_dir, clean_dir):
        d.mkdir(exist_ok=True)

    NEG_BASE = (
        "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
        "heavy black outline, sticker, cross-hatching, pen hatching, "
        "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
        "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
        "multiple animals, duplicate, tiling, text, watermark, border, cropped body, cut off, "
        "box turtle, speckled background, confetti"
    )
    STYLE_CLAUSE = (
        "rich natural colors, crisp distinct painted color steps, flat stylized "
        "gouache texture marks, clean thin dark charcoal-umber outline, "
        "isolated on a plain solid white background"
    )

    prompts = []
    descriptions = {}
    for i, (inp_stem, slug, common, wiki_title, body_clause) in enumerate(SPECIES):
        src_cut = WORK / "inputs" / f"{inp_stem}.png"
        shutil.copyfile(src_cut, in_dir / f"{i:02d}_{slug}.png")

        wiki_txt = fetch_full_wiki_description(wiki_title)
        desc = call_gemma27b_rich(src_cut, common, wiki_txt)
        descriptions[slug] = desc
        print(f"\n[{slug}] ({wiki_title}):\n  {desc}", flush=True)

        article = "an" if common[0].lower() in "aeiou" else "a"
        p = (
            f"{article} {common.lower()}, {desc}, "
            f"full body wildlife illustration, single animal {body_clause}, "
            f"{STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        )
        prompts.append(p)

    (RICH_DIR / "rich_descriptions.json").write_text(json.dumps(descriptions, indent=2), encoding="utf-8")
    pfile = RICH_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    # Free gemma3:27b from VRAM before running SDXL
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
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    cols = [
        "1. Input Cutout",
        "2. No Color Hint (Box Turtle Overfit)",
        "3. Too-Short 8w Hint (All Green)",
        "4. Rich Wiki+Cutout gemma3:27b (22w)",
    ]
    cw, ch = 360, 360
    W = len(cols) * cw
    H = 48 + len(SPECIES) * (ch + 36)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 12, 12), t, fill=(20, 20, 20), font=font)

    for ri, (inp_stem, slug, common, _, _) in enumerate(SPECIES):
        y_top = 48 + ri * (ch + 36)
        d.rectangle([0, y_top, W, y_top + 36], fill=(232, 232, 236))
        desc_short = descriptions[slug][:125] + ("..." if len(descriptions[slug]) > 125 else "")
        d.text((12, y_top + 8), f"{common}: {desc_short}", fill=(10, 10, 10), font=font_sm)
        files = [
            WORK / "inputs" / f"{inp_stem}.png",
            WORK / "clean_str_0.76" / f"{inp_stem}.png",
            WORK / "controlled_s1250_test" / "clean_0.76" / f"{ri:02d}_{slug}.png",
            clean_dir / f"{ri:02d}_{slug}.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 36))
            d.rectangle([ci * cw, y_top + 36, (ci + 1) * cw - 1, y_top + 36 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "rich_wiki_gemma27b_compare.jpg", quality=92)
    print("Saved rich_wiki_gemma27b_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
