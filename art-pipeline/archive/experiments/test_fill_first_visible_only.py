"""Test the clean, zero-pose, VLM-driven prompt architecture:
1. Pass full Wikipedia Description + Cutout to gemma3:27b.
2. Ask gemma3:27b for ONLY two things:
   a) `missing_parts_to_fill`: ONLY what is visibly cut off / missing at the bottom of the cutout (e.g. 'with all four slender legs and dark cloven hooves visible', 'with webbed feet visible underneath', 'with rounded hind haunch and four small paws visible', or '' if nothing is cut off!).
   b) `visible_color_and_markings`: 15-20 word description of ONLY the body parts & markings actually visible from this angle (using Wikipedia for exact species marking names, and skipping invisible parts like plastron/underwing/ventral scales).
3. Put `missing_parts_to_fill` FIRST right after the species name (so SDXL completes the cut-off anatomy with full token weight), followed by `visible_color_and_markings` and `STYLE_CLAUSE` — with ZERO pose descriptions!
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
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

OLLAMA = "http://localhost:11434"
LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
ZERO_POSE_DIR = WORK / "zero_pose_vlm_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SPECIES = [
    ("04_mule_deer", "mule_deer", "Mule Deer", "Mule deer"),
    ("08_pond_slider", "pond_slider", "Pond Slider", "Red-eared slider"),
    ("03_green-winged_teal", "green-winged_teal", "Green-winged Teal", "Green-winged teal"),
    ("07_american_pika", "american_pika", "American Pika", "American pika"),
    ("06_california_ground_squirrel", "california_ground_squirrel", "California Ground Squirrel", "California ground squirrel"),
    ("09_common_watersnake", "common_watersnake", "Common Watersnake", "Common watersnake"),
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
   - If the animal's lower legs, hooves, webbed feet, paws, or tail are cut off or hidden (e.g. by grass, water, or a rock that was removed), write a short 4-8 word phrase telling the illustrator what missing anatomy to draw in (e.g. 'with all four slender legs and dark hooves visible', 'with webbed feet visible underneath', 'with rounded hind haunch and four paws visible', 'with webbed swimming feet visible').
   - If the animal's body and limbs/coils are already complete (like a coiled snake or a standing squirrel with paws visible), return an empty string `""`.
2. `visible_color_and_markings`: Using the Wikipedia Description for accurate species marking names, write a concise 14-20 word comma-separated list of `[color/pattern] [body part]` phrases for ONLY the parts of the animal that are visible from this camera angle.
   - DO NOT mention hidden underside parts that cannot be seen from this angle (no 'plastron', 'ventral scales', 'underwing coverts', 'black toe pads').
   - DO NOT describe pose, posture, or camera angle (no 'standing', 'coiled', 'floating', 'side view').
   - DO NOT mention environment, water, grass, rocks, or logs.
   - DO NOT use the word 'pinkish'.

Return STRICT JSON only:
{{
  "missing_parts_to_fill": "short 4-8 word phrase for cut-off anatomy, or empty string",
  "visible_color_and_markings": "14-20 word comma-separated list of visible `[color/pattern] [body part]` phrases"
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
    ZERO_POSE_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = ZERO_POSE_DIR / "inputs"
    out_dir = ZERO_POSE_DIR / "outputs"
    clean_dir = ZERO_POSE_DIR / "clean"
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
    for i, (inp_stem, slug, common, wiki_title) in enumerate(SPECIES):
        src_cut = WORK / "inputs" / f"{inp_stem}.png"
        dst_cut = in_dir / f"{i:02d}_{slug}.png"
        shutil.copyfile(src_cut, dst_cut)

        wiki_txt = fetch_full_wiki_description(wiki_title)
        res = call_gemma27b(dst_cut, common, wiki_txt)
        vlm_out[slug] = res
        print(f"\n[{slug}]:\n  missing_parts_to_fill: {res.get('missing_parts_to_fill')!r}\n  visible_color_and_markings: {res.get('visible_color_and_markings')!r}", flush=True)

        article = "an" if common[0].lower() in "aeiou" else "a"
        missing = res.get("missing_parts_to_fill", "").strip()
        colors = res.get("visible_color_and_markings", "").strip()

        # Put missing_parts_to_fill FIRST (right after species name), ZERO pose description, then visible colors + style!
        if missing:
            p = f"{article} {common.lower()} {missing}, full body wildlife illustration, single animal, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        else:
            p = f"{article} {common.lower()}, full body wildlife illustration, single animal, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        prompts.append(p)

    (ZERO_POSE_DIR / "vlm_outputs.json").write_text(json.dumps(vlm_out, indent=2), encoding="utf-8")
    pfile = ZERO_POSE_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    # Unload gemma3:27b from VRAM before SDXL
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
        "1. Input Cutout",
        "2. Old Baseline (No Wiki/VLM)",
        "3. Prev Run (35w + Invisible Parts First)",
        "4. New: Fill-First + Visible-Only (Zero Pose)",
    ]
    cw, ch = 360, 360
    W = len(cols) * cw
    H = 48 + len(SPECIES) * (ch + 44)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 12, 12), t, fill=(20, 20, 20), font=font)

    for ri, (inp_stem, slug, common, _) in enumerate(SPECIES):
        y_top = 48 + ri * (ch + 44)
        d.rectangle([0, y_top, W, y_top + 44], fill=(232, 232, 236))
        m_txt = vlm_out[slug].get("missing_parts_to_fill", "") or "(none)"
        c_txt = vlm_out[slug].get("visible_color_and_markings", "")
        d.text((12, y_top + 4), f"{common} | Fill: {m_txt}", fill=(10, 10, 80), font=font_sm)
        d.text((12, y_top + 23), f"Visible Colors: {c_txt[:135]}", fill=(30, 30, 30), font=font_sm)

        files = [
            WORK / "inputs" / f"{inp_stem}.png",
            WORK / "clean_str_0.76" / f"{inp_stem}.png",
            WORK / "rich_wiki_vlm_test" / "clean" / f"{ri:02d}_{slug}.png",
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

    sheet.save(ARTIFACT_DIR / "fill_first_visible_only_compare.jpg", quality=92)
    print("Saved fill_first_visible_only_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
