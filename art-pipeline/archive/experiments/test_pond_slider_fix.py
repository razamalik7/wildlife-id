"""Quick verification of Pond Slider fix:
1. Updated PROMPT_TEMPLATE where base color comes from image pixels first + Wikipedia variant markings.
2. Remove the artificial bottom-strip blur extension on non-ungulate modes so the rear of the turtle shell isn't stretched into a false head.
3. Compare strengths 0.68, 0.72, 0.75 on Pond Slider.
"""

import base64
import json
import subprocess
import time
from io import BytesIO
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from u2net_matte import remove
from normalize import subject_bbox
from test_birefnet_variant_vlm import fetch_full_wiki_description

OLLAMA = "http://localhost:11434"
LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
WORK = Path(__file__).parent / "batch_run_v4"
BIREF_TEST = WORK / "birefnet_vlm_test"
PS_TEST = WORK / "pond_slider_fix_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity/brain/cf192390-434f-49ae-a401-59243749f021")

PROMPT_TEMPLATE_V2 = """You are a wildlife illustrator inspecting a cutout image of a {common_name} alongside its Wikipedia Description.

Wikipedia Description for {common_name}:
\"\"\"{wiki_text}\"\"\"

Your job is to return ONLY two short fields in JSON:
1. `missing_parts_to_fill`: Look at the bottom edge of the animal cutout.
   - If the animal's lower legs, hooves, or webbed feet are completely cut off at a flat horizontal bottom line (by water, grass, or rock):
     * For standing hoofed mammals (deer, elk, goats, etc.) whose lower legs are cut off, write: `"standing on all four slender legs with dark cloven hooves visible"`
     * For floating/standing ducks or waterbirds whose feet are hidden underwater, write: `"with webbed feet visible underneath"`
     * For small crouching mammals on rocks whose bottom paws are cut off, write: `"with rounded hind haunch and paws visible"`
   - If the animal already has legs/feet/coils visible (such as a turtle with legs visible, a squirrel with paws visible, or a coiled snake), return `""`.
2. `visible_color_and_markings`:
   - RULE 1 (BASE COLOR FROM IMAGE FIRST): Look directly at the attached image first to determine the actual base color of the animal's shell, fur, scales, or plumage in this photo. When there are age, sex, or regional variants listed in the Wikipedia description (e.g., a brownish-olive adult turtle shell vs a dark green young shell, or breeding male vs female plumage), determine WHICH variant matches the image pixels and use THAT variant's base color (for example, if the turtle shell in the image is brown / olive-brown, you MUST write `"brownish-olive oval carapace"` and NEVER `"dark green carapace"`).
   - RULE 2 (DETAILED MARKINGS FROM WIKIPEDIA): Combine that true base color with the species' diagnostic visible markings from the Wikipedia description (for example: `"brownish-olive oval carapace with yellow scute markings, dark green head and limbs with fine yellow stripes, distinct bright red stripe behind eye"`).
   - Write a 16-24 word comma-separated list of `[color/pattern] [body part]` phrases for ONLY the parts of the animal visible from this angle.
   - DO NOT mention hidden underside parts (no 'plastron', 'ventral scales', 'underwing coverts').
   - DO NOT describe pose or camera angle.
   - DO NOT mention environment, water, grass, rocks, or logs.
   - DO NOT use the word 'pinkish'.

Return STRICT JSON only:
{{
  "missing_parts_to_fill": "short phrase for cut-off anatomy, or empty string",
  "visible_color_and_markings": "16-24 word comma-separated list of visible `[color/pattern] [body part]` phrases matching the variant in the image"
}}"""


def call_gemma27b_v2(cut_p: Path, common_name: str, wiki_text: str) -> dict:
    im = Image.open(cut_p).convert("RGB")
    im.thumbnail((768, 768), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = PROMPT_TEMPLATE_V2.format(common_name=common_name, wiki_text=wiki_text)
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


def main() -> None:
    PS_TEST.mkdir(parents=True, exist_ok=True)
    in_dir = PS_TEST / "inputs"
    out_dir = PS_TEST / "outputs"
    clean_dir = PS_TEST / "clean"
    for d in (in_dir, out_dir, clean_dir):
        d.mkdir(exist_ok=True)

    biref_cut = BIREF_TEST / "pond_slider_biref_cutout.png"
    # Notice: in pond_slider_biref_cutout.png, the bottom-right has a leftover piece of branch (x>680, y>740).
    # Let's see what happens both with raw biref_cut and with the bottom-right branch stub cleaned!
    im = Image.open(biref_cut).convert("RGB")
    im.save(in_dir / "00_pond_slider_raw_biref.png")

    # Also test trimming the leftover branch stub at bottom-right (below the rear shell rim)
    arr = np.array(im).copy()
    # In pond_slider_biref_cutout.png, the rear shell edge ends around y=810 on the right, and x=680..885, y=735..870 is the branch stub sticking out
    # Let's create a clean version where that bottom-right branch stub is masked out along the elliptical shell arc
    yy, xx = np.mgrid[:1024, :1024]
    # Ellipse approximating the turtle shell + head + left legs, excluding the bottom-right branch protrusion
    branch_stub = (xx > 690) & (yy > 745) & ((xx - 520) ** 2 / (270 ** 2) + (yy - 560) ** 2 / (245 ** 2) > 1.0)
    arr[branch_stub] = 255
    im_trimmed = Image.fromarray(arr)
    im_trimmed.save(in_dir / "01_pond_slider_trimmed_biref.png")

    wiki_txt = fetch_full_wiki_description("Red-eared slider")
    res = call_gemma27b_v2(in_dir / "01_pond_slider_trimmed_biref.png", "Pond Slider", wiki_txt)
    print("Gemma 27B v2 output for Pond Slider:", json.dumps(res, indent=2), flush=True)

    requests.post(f"{OLLAMA}/api/generate", json={"model": "gemma3:27b", "keep_alive": 0}, timeout=15)
    time.sleep(2)

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

    colors = res.get("visible_color_and_markings", "").strip()
    missing = res.get("missing_parts_to_fill", "").strip()
    if missing:
        prompt_str = f"a pond slider, full body wildlife illustration, single animal {missing}, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"
    else:
        prompt_str = f"a pond slider, full body wildlife illustration, single animal, {colors}, {STYLE_CLAUSE} --n {NEG_BASE} --d 42"

    pfile = PS_TEST / "prompts.txt"
    pfile.write_text("\n".join([prompt_str, prompt_str]), encoding="utf-8")

    for strength in ["0.70", "0.74"]:
        s_out = out_dir / f"s_{strength}"
        s_out.mkdir(exist_ok=True)
        subprocess.run([
            str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
            "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
            "--outdir", str(s_out),
            "--network_module", "networks.lora",
            "--network_weights", str(LORA / "output" / "animl_style_v4-step00001250.safetensors"),
            "--network_mul", "1.0",
            "--max_embeddings_multiples", "3",
            "--from_file", str(pfile),
            "--image_path", str(in_dir),
            "--use_original_file_name",
            "--strength", strength,
            "--W", "1024", "--H", "1024",
            "--steps", "28", "--sampler", "euler_a",
            "--scale", "7.0", "--bf16", "--sdpa",
            "--batch_size", "1", "--images_per_prompt", "1",
        ], check=True)

    # Build quick comparison sheet
    cw, ch = 360, 360
    sheet = Image.new("RGB", (cw * 4, ch + 70), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arialbd.ttf", 18)
        font_sm = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    d.text((12, 8), f"Pond Slider V2 Prompt Colors: {colors}", fill=(10, 10, 80), font=font_sm)
    titles = ["1. Raw BiRefNet (No Smudge, s=0.70)", "2. Raw BiRefNet (No Smudge, s=0.74)", "3. Trimmed Stub (s=0.70)", "4. Trimmed Stub (s=0.74)"]
    files = [
        out_dir / "s_0.70" / "00_pond_slider_raw_biref.png",
        out_dir / "s_0.74" / "00_pond_slider_raw_biref.png",
        out_dir / "s_0.70" / "01_pond_slider_trimmed_biref.png",
        out_dir / "s_0.74" / "01_pond_slider_trimmed_biref.png",
    ]
    for ci, (t, fp) in enumerate(zip(titles, files)):
        d.text((ci * cw + 10, 36), t, fill=(20, 20, 20), font=font)
        im_out = Image.open(fp).convert("RGBA")
        cut = remove(im_out)
        box = subject_bbox(cut)
        subj = cut.crop(box)
        scale = int((cw - 16) * 0.85) / max(subj.size)
        nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
        subj = subj.resize((nw, nh), Image.LANCZOS)
        cell = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        cell.paste(subj, ((cw - nw) // 2, (ch - nh) // 2), subj)
        sheet.paste(cell.convert("RGB"), (ci * cw, 64))
        d.rectangle([ci * cw, 64, (ci + 1) * cw - 1, 64 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "pond_slider_fix_compare.jpg", quality=92)
    print("Saved pond_slider_fix_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
