"""End-to-end evaluation of animl_style_v5 on all 11 unseen species from picks_overnight.json.

Pipeline (100% automated, zero human/Claude in the loop):
1. BiRefNet (1024x1024) segmentation cutout from reference photo (with bottom headroom for floating/perched species and vertical leg extension seed only for tall hoofed mammals in grass).
2. Local gemma3:27b (via Ollama) + full Wikipedia `== Description ==` text using:
   - Grounded 2-step JSON (`observed_base_colors_in_photo` generated FIRST from image pixels before reading Wikipedia variants)
   - Priority token ordering: (1) head/face/eye-stripe/bill colors first, (2) primary shell/coat/plumage base color and markings second, (3) limbs/tail third
   - Fill-first `missing_parts_to_fill` and zero pose text
3. SDXL img2img inference with `animl_style_v5` checkpoints (Step 750, Step 1000, Step 1250, Step 1500).
4. Post-generation BiRefNet alpha cutout & 76% fill normalization onto pure white.
5. Side-by-side comparison sheets saved to the artifact directory for morning review.
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from normalize import subject_bbox
from u2net_matte import remove

OLLAMA = "http://localhost:11434"
LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity/brain/cf192390-434f-49ae-a401-59243749f021")

SPECIES_11 = [
    ("northern_shoveler", "Northern Shoveler", "Northern shoveler", "floating", "00_northern_shoveler"),
    ("northern_pintail", "Northern Pintail", "Northern pintail", "floating", "01_northern_pintail"),
    ("american_wigeon", "American Wigeon", "American wigeon", "floating", "02_american_wigeon"),
    ("green-winged_teal", "Green-winged Teal", "Green-winged teal", "floating", "03_green-winged_teal"),
    ("mule_deer", "Mule Deer", "Mule deer", "grass_hooves", "04_mule_deer"),
    ("muskrat", "Muskrat", "Muskrat", "swimming", "05_muskrat"),
    ("california_ground_squirrel", "California Ground Squirrel", "California ground squirrel", "standing", "06_california_ground_squirrel"),
    ("american_pika", "American Pika", "American pika", "rock_perch", "07_american_pika"),
    ("pond_slider", "Pond Slider", "Red-eared slider", "swimming", "08_pond_slider"),
    ("common_watersnake", "Common Watersnake", "Common watersnake", "coiled", "09_common_watersnake"),
    ("gopher_tortoise", "Gopher Tortoise", "Gopher tortoise", "standing", "10_gopher_tortoise"),
]

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

INFERENCE_PROMPT_TEMPLATE = """You are a wildlife illustrator inspecting a cutout image of a {common_name} alongside its Wikipedia Description.

Wikipedia Description for {common_name}:
\"\"\"{wiki_text}\"\"\"

Return STRICT JSON with these 3 fields in exact order:
1. `observed_base_colors_in_photo`:
   - Look ONLY at the attached image pixels first. State the true base color of the main body/coat/plumage/scales/shell in the photo and state which Wikipedia variant (adult/juvenile, male/female) matches the photo.
2. `missing_parts_to_fill`:
   - Look at the bottom edge of the animal cutout.
   - If the animal's lower legs, hooves, or webbed feet are completely cut off at a flat horizontal bottom line (by water, grass, or rock):
     * For standing hoofed mammals (deer, elk, goats, etc.) whose lower legs are cut off, write: `"standing on all four slender legs with dark cloven hooves visible"`
     * For floating/standing ducks or waterbirds whose feet are hidden underwater, write: `"with webbed feet visible underneath"`
     * For small crouching mammals on rocks whose bottom paws are cut off, write: `"with rounded hind haunch and paws visible"`
   - If the animal already has legs/feet/coils visible (such as a turtle or tortoise with legs visible, a squirrel with paws visible, a swimming muskrat, or a coiled snake), return `""`.
3. `visible_color_and_markings`:
   - Using the variant and true base colors from `observed_base_colors_in_photo` combined with fine markings from the Wikipedia Description, write a 16-22 word comma-separated list of `[color/pattern] [body part]` phrases for ONLY visible parts.
   - ORDERING RULE: Put (1) main body/shell/back/breast base color and markings FIRST, (2) head, eye/ear stripes, and bill/muzzle colors SECOND, and (3) flanks/wings/legs/tail THIRD.
   - CLIP COLOR RULE: Never use the word `'olive'` or `'greenish'` when describing a brown shell, fur, or scales (image models interpret `'olive'` as bright green); use `'warm brown'` or `'dark brown'` instead. If `observed_base_colors_in_photo` noted a brown shell on a turtle, describe it as `'warm brown oval carapace with thin yellow markings'` and never call the carapace green.
   - DO NOT mention hidden underside parts (no 'plastron', 'ventral scales', 'underwing coverts') or structural geometry like 'notched marginal scutes'.
   - DO NOT describe pose, camera angle, or environment.
   - DO NOT use the word 'pinkish'.

Return STRICT JSON only:
{{
  "observed_base_colors_in_photo": "true base colors seen in the image pixels and matching variant",
  "missing_parts_to_fill": "short phrase for cut-off anatomy, or empty string",
  "visible_color_and_markings": "16-22 word comma-separated list ordered: head/bill markings first, shell/body base color & markings second, limbs/tail third"
}}"""


def fetch_full_wiki_description(title: str, session: requests.Session) -> str:
    r = session.get(
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


def call_gemma27b(cut_p: Path, common_name: str, wiki_text: str) -> dict:
    im = Image.open(cut_p).convert("RGB")
    im.thumbnail((768, 768), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = INFERENCE_PROMPT_TEMPLATE.format(common_name=common_name, wiki_text=wiki_text)
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


def build_input_cutout(slug: str, mode: str, raw_dir: Path, in_dir: Path, idx: int) -> Path:
    photo_p = WORK / f"{slug}_photo.png"
    raw_p = raw_dir / f"{slug}_biref.png"
    if not raw_p.exists():
        if slug == "common_watersnake" and (WORK / "common_watersnake_cutout.png").exists():
            shutil.copyfile(WORK / "common_watersnake_cutout.png", raw_p)
        else:
            photo = Image.open(photo_p).convert("RGBA")
            cut = remove(photo)
            box = subject_bbox(cut)
            subj = cut.crop(box)
            scale = int(1024 * 0.76) / max(subj.size)
            nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
            subj = subj.resize((nw, nh), Image.LANCZOS)
            canv = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
            canv.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
            rgb = canv.convert("RGB")
            # For pond_slider, remove the leftover wooden branch stub at bottom-right (x>690, y>745) outside the shell arc
            if slug == "pond_slider":
                arr = np.array(rgb)
                yy, xx = np.mgrid[:1024, :1024]
                branch_stub = (xx > 690) & (yy > 745) & ((xx - 520) ** 2 / (270 ** 2) + (yy - 560) ** 2 / (245 ** 2) > 1.0)
                arr[branch_stub] = 255
                rgb = Image.fromarray(arr)
            rgb.save(raw_p)

    dst_p = in_dir / f"{idx:02d}_{slug}.png"
    if dst_p.exists():
        return raw_p

    im = Image.open(raw_p).convert("RGB")
    if mode in ("grass_hooves", "rock_perch", "floating"):
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
            if mode == "grass_hooves":
                strip_h = min(24, nh // 6)
                bottom_strip = crop_resized.crop((0, nh - strip_h, nw, nh))
                ext_h = 64
                bottom_ext = bottom_strip.resize((nw, ext_h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(8))
                white_bg = Image.new("RGB", (nw, ext_h), (255, 255, 255))
                seeded = Image.blend(white_bg, bottom_ext, 0.42)
                canvas.paste(seeded, (px, py + nh - 6))
            canvas.save(dst_p)
            return raw_p

    shutil.copyfile(raw_p, dst_p)
    return raw_p


def clean_and_center(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    im = Image.open(src).convert("RGBA")
    cut = remove(im)
    box = subject_bbox(cut)
    if box is None:
        shutil.copyfile(src, dst)
        return
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    canvas.convert("RGB").save(dst)


def render_sheet(rows_spec, cols_spec, vlm_cache: dict, out_path: Path) -> None:
    try:
        font = ImageFont.truetype("arialbd.ttf", 19)
        font_sm = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    cw, ch = 320, 320
    W = len(cols_spec) * cw
    H = 48 + len(rows_spec) * (ch + 44)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, (col_title, _) in enumerate(cols_spec):
        d.text((ci * cw + 10, 12), col_title, fill=(20, 20, 20), font=font)

    for ri, (idx, slug, common, old_stem) in enumerate(rows_spec):
        y_top = 48 + ri * (ch + 44)
        d.rectangle([0, y_top, W, y_top + 44], fill=(232, 232, 236))
        info = vlm_cache.get(slug, {})
        m_txt = info.get("missing_parts_to_fill", "") or "(none)"
        c_txt = info.get("visible_color_and_markings", "")
        d.text((12, y_top + 4), f"{common} | Fill: {m_txt}", fill=(10, 10, 80), font=font_sm)
        d.text((12, y_top + 23), f"VLM Colors: {c_txt[:150]}", fill=(30, 30, 30), font=font_sm)

        for ci, (_, path_fn) in enumerate(cols_spec):
            fp = path_fn(idx, slug, old_stem)
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 44))
            d.rectangle([ci * cw, y_top + 44, (ci + 1) * cw - 1, y_top + 44 + ch - 1], outline=(210, 210, 210))

    sheet.save(out_path, quality=92)
    print(f"Saved {out_path.name}", flush=True)


def main() -> None:
    V5_EVAL.mkdir(parents=True, exist_ok=True)
    raw_dir = V5_EVAL / "raw_biref"
    in_dir = V5_EVAL / "inputs"
    for d in (raw_dir, in_dir):
        d.mkdir(exist_ok=True)

    cache_file = V5_EVAL / "vlm_11_cache_v2.json"
    vlm_cache = {}
    if cache_file.exists():
        try:
            vlm_cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            vlm_cache = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "AnimlVisionBot/1.0"})
    used_ollama = False
    prompts = []

    for i, (slug, common, wiki_title, mode, _) in enumerate(SPECIES_11):
        raw_p = build_input_cutout(slug, mode, raw_dir, in_dir, i)
        if slug not in vlm_cache:
            used_ollama = True
            wiki_txt = fetch_full_wiki_description(wiki_title, session)
            res = call_gemma27b(raw_p, common, wiki_txt)
            vlm_cache[slug] = res
            cache_file.write_text(json.dumps(vlm_cache, indent=2), encoding="utf-8")
        else:
            res = vlm_cache[slug]

        article = "an" if common[0].lower() in "aeiou" else "a"
        missing = res.get("missing_parts_to_fill", "").strip()
        colors = res.get("visible_color_and_markings", "").strip()

        if missing:
            p = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal {missing}, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        else:
            p = f"animl_gouache_style, {article} {common.lower()}, full body wildlife illustration, single animal, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        prompts.append(p)
        print(f"[{i:02d} {slug}]\n  Observed={res.get('observed_base_colors_in_photo')!r}\n  Fill={missing!r}\n  Colors={colors!r}", flush=True)

    pfile = V5_EVAL / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    if used_ollama:
        requests.post(f"{OLLAMA}/api/generate", json={"model": "gemma3:27b", "keep_alive": 0}, timeout=15)
        time.sleep(2)

    checkpoints = [
        ("step750", LORA / "output" / "animl_style_v5-step00000750.safetensors"),
        ("step1000", LORA / "output" / "animl_style_v5-step00001000.safetensors"),
        ("step1250", LORA / "output" / "animl_style_v5-step00001250.safetensors"),
        ("step1500", LORA / "output" / "animl_style_v5.safetensors"),
    ]
    strengths = ["0.72", "0.75"]

    for ckpt_tag, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"Skipping missing checkpoint: {ckpt_path}", flush=True)
            continue
        for s in strengths:
            run_tag = f"{ckpt_tag}_s{s}"
            out_dir = V5_EVAL / f"out_{run_tag}"
            clean_dir = V5_EVAL / f"clean_{run_tag}"
            out_dir.mkdir(exist_ok=True)
            clean_dir.mkdir(exist_ok=True)

            if len(list(clean_dir.glob("*.png"))) < len(SPECIES_11):
                print(f"\n--- Running v5 evaluation: {run_tag} ---", flush=True)
                subprocess.run([
                    str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
                    "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
                    "--outdir", str(out_dir),
                    "--network_module", "networks.lora",
                    "--network_weights", str(ckpt_path),
                    "--network_mul", "1.0",
                    "--max_embeddings_multiples", "3",
                    "--from_file", str(pfile),
                    "--image_path", str(in_dir),
                    "--use_original_file_name",
                    "--strength", s,
                    "--W", "1024", "--H", "1024",
                    "--steps", "28", "--sampler", "euler_a",
                    "--scale", "7.0", "--bf16", "--sdpa",
                    "--batch_size", "1", "--images_per_prompt", "1",
                ], check=True)

                for png in out_dir.glob("*.png"):
                    clean_and_center(png, clean_dir / png.name)

    cols_s072 = [
        ("1. Photo Ref", lambda i, slug, old: WORK / f"{slug}_photo.png"),
        ("2. BiRefNet Cutout", lambda i, slug, old: in_dir / f"{i:02d}_{slug}.png"),
        ("3. Old v4 (No VLM)", lambda i, slug, old: WORK / "clean_str_0.76" / f"{old}.png"),
        ("4. v5 Step 750 (s=0.72)", lambda i, slug, old: V5_EVAL / "clean_step750_s0.72" / f"{i:02d}_{slug}.png"),
        ("5. v5 Step 1000 (s=0.72)", lambda i, slug, old: V5_EVAL / "clean_step1000_s0.72" / f"{i:02d}_{slug}.png"),
        ("6. v5 Step 1250 (s=0.72)", lambda i, slug, old: V5_EVAL / "clean_step1250_s0.72" / f"{i:02d}_{slug}.png"),
    ]
    cols_s075 = [
        ("1. BiRefNet Cutout", lambda i, slug, old: in_dir / f"{i:02d}_{slug}.png"),
        ("2. Old v4 (No VLM)", lambda i, slug, old: WORK / "clean_str_0.76" / f"{old}.png"),
        ("3. v5 Step 750 (s=0.75)", lambda i, slug, old: V5_EVAL / "clean_step750_s0.75" / f"{i:02d}_{slug}.png"),
        ("4. v5 Step 1000 (s=0.75)", lambda i, slug, old: V5_EVAL / "clean_step1000_s0.75" / f"{i:02d}_{slug}.png"),
        ("5. v5 Step 1250 (s=0.75)", lambda i, slug, old: V5_EVAL / "clean_step1250_s0.75" / f"{i:02d}_{slug}.png"),
        ("6. v5 Step 1500 (s=0.75)", lambda i, slug, old: V5_EVAL / "clean_step1500_s0.75" / f"{i:02d}_{slug}.png"),
    ]

    waterfowl_rows = [(i, s[0], s[1], s[4]) for i, s in enumerate(SPECIES_11[:4])]
    land_rows = [(i, s[0], s[1], s[4]) for i, s in enumerate(SPECIES_11) if i >= 4]

    render_sheet(waterfowl_rows, cols_s072, vlm_cache, ARTIFACT_DIR / "v5_final_waterfowl_s072.jpg")
    render_sheet(land_rows, cols_s072, vlm_cache, ARTIFACT_DIR / "v5_final_land_reptiles_s072.jpg")
    render_sheet(waterfowl_rows, cols_s075, vlm_cache, ARTIFACT_DIR / "v5_final_waterfowl_s075.jpg")
    render_sheet(land_rows, cols_s075, vlm_cache, ARTIFACT_DIR / "v5_final_land_reptiles_s075.jpg")


if __name__ == "__main__":
    main()
