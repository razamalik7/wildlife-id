"""End-to-end 5-species test of gemma3:27b inspecting the white-background cutout
(+ Wikipedia reference snippet) -> generating the prompt + completion_mode ->
applying the matching bottom-edge seed -> running SDXL img2img.
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
E2E_DIR = WORK / "gemma27b_cutout_test"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

TEST_CASES = [
    ("mule_deer", "Mule Deer", WORK / "mule_deer_cutout.png",
     "Mule deer have large ears, a tan-brown coat, white rump patch, black-tipped tail, forked antlers in males, and slender legs with dark cloven hooves."),
    ("pond_slider", "Pond Slider", WORK / "pond_slider_cutout.png",
     "Adult pond sliders have an oval olive-to-dark-brown carapace, yellow and green striped skin on head and legs, a red stripe behind the eye, and webbed clawed feet."),
    ("green-winged_teal", "Green-winged Teal", WORK / "green-winged_teal_cutout.png",
     "Breeding male green-winged teal have a chestnut head with green eye-patch, speckled buff breast, finely barred grey flanks, yellow undertail patch, and dark grey-brown webbed feet."),
    ("american_pika", "American Pika", WORK / "american_pika_cutout.png",
     "American pikas are small, round, tailless lagomorphs with grizzled greyish-brown and buff fur, rounded ears, and four short furry legs with small paws."),
    ("california_ground_squirrel", "California Ground Squirrel", WORK / "california_ground_squirrel_cutout.png",
     "California ground squirrels have mottled brown and grey fur with a pale grey shoulder cape, buffy belly, bushy tail, and four clawed paws."),
]

PROMPT_TEMPLATE = """You are a wildlife illustration director inspecting a white-background cutout of a {common_name}.
Background species reference: {wiki_hint}

CRITICAL RULES:
1. Describe ONLY the animal's body, fur/feathers/scales/shell, and anatomy in `[color] [body part]` pairs (e.g. 'dark brown carapace', 'chestnut head', 'emerald green eye-patch').
2. NEVER mention any environment, background, water, grass, rocks, logs, or branches. If a piece of a wooden log or rock is stuck to the cutout, ignore it.
3. Describe ONLY the true colors visible on THIS individual animal (e.g., if an adult turtle shell is dark brown/black with subtle olive scutes, do NOT say yellow lines or bright green shell; never use bleeding words like 'pinkish').
4. Look closely at the bottom edge of the animal cutout: are the lower legs/hooves, webbed feet, or hind paws cut off or missing? Pick the matching `completion_mode`.

Return STRICT JSON only with these exact fields:
{{
  "pose_and_view": "short phrase describing the animal's pose and angle (e.g. 'standing side three-quarter view', 'floating side profile view', 'crouching side three-quarter view')",
  "visible_colors_and_markings": "15-25 word comma-separated list of `[color] [body part]` phrases describing ONLY the animal's visible coloration and markings.",
  "missing_or_cutoff_parts": "which lower body parts are cut off or missing in this cutout (e.g. 'lower legs and cloven hooves cut off at bottom', 'webbed feet missing below waterline', 'hind haunch and hind paws sliced diagonally', 'rear swimming foot hidden', or 'none')",
  "completed_anatomy_clause": "short phrase describing how the full body and completed feet/legs should appear (e.g. 'complete full body with all four slender legs and dark cloven hooves visible', 'complete full body with dark grey webbed feet tucked beneath belly', 'complete full body with rounded hind haunch and all four small paws visible', 'complete full body with outstretched swimming legs and webbed clawed feet visible')",
  "completion_mode": "one of: ['none', 'extend_hooves_down', 'duck_feet_tucked', 'duck_feet_paddling', 'round_bottom_and_paws', 'remove_log_and_swim']"
}}"""


def call_gemma27b(im_path: Path, common_name: str, wiki_hint: str) -> dict:
    im = Image.open(im_path).convert("RGB")
    im.thumbnail((768, 768), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = PROMPT_TEMPLATE.format(common_name=common_name, wiki_hint=wiki_hint)
    t0 = time.time()
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
    dt = time.time() - t0
    r.raise_for_status()
    res = json.loads(r.json()["message"]["content"])
    res["_sec"] = round(dt, 1)
    return res


def prepare_seeded_input(src_cut: Path, dst_p: Path, mode: str) -> None:
    """Apply deterministic latent seeding based on gemma3:27b's completion_mode."""
    im = Image.open(src_cut).convert("RGB")
    arr = np.array(im)
    mask = (arr < 245).any(axis=2)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        im.save(dst_p)
        return
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = im.crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = crop.size

    if mode == "extend_hooves_down":
        # Shift up and extend four leg-colored vertical smudges down so SDXL can complete slender legs + hooves even at 0.68-0.72!
        scale = int(1024 * 0.66) / max(cw, ch)
        nw, nh = round(cw * scale), round(ch * scale)
        crop_r = crop.resize((nw, nh), Image.LANCZOS)
        canv = Image.new("RGB", (1024, 1024), (255, 255, 255))
        px = (1024 - nw) // 2
        py = max(36, (1024 - nh) // 2 - 55)
        canv.paste(crop_r, (px, py))
        # Extrude the bottom 28px strip downward by 70px to extend the 4 leg stumps!
        strip_h = 28
        bottom_strip = crop_r.crop((0, nh - strip_h, nw, nh))
        ext_h = 72
        bottom_ext = bottom_strip.resize((nw, ext_h), Image.NEAREST).filter(ImageFilter.GaussianBlur(4))
        white_bg = Image.new("RGB", (nw, ext_h), (255, 255, 255))
        seeded = Image.blend(white_bg, bottom_ext, 0.72)
        canv.paste(seeded, (px, py + nh - 8))
        canv.save(dst_p)
    elif mode in ("duck_feet_tucked", "duck_feet_paddling"):
        scale = int(1024 * 0.68) / max(cw, ch)
        nw, nh = round(cw * scale), round(ch * scale)
        crop_r = crop.resize((nw, nh), Image.LANCZOS)
        canv = Image.new("RGB", (1024, 1024), (255, 255, 255))
        px = (1024 - nw) // 2
        py = (1024 - nh) // 2 - 35
        canv.paste(crop_r, (px, py))
        d = ImageDraw.Draw(canv)
        by = py + nh - 12
        belly_col = crop_r.getpixel((nw // 2, max(0, nh - 20)))
        d.chord([px + int(nw * 0.16), by - 26, px + int(nw * 0.78), by + 26], start=0, end=180, fill=belly_col)
        tx, ty = px + int(nw * 0.34), by + 8
        d.polygon([(tx, ty), (tx - 55, ty + 14), (tx - 42, ty + 36), (tx + 8, ty + 22)], fill=(125, 128, 132), outline=(55, 50, 48))
        canv = canv.filter(ImageFilter.GaussianBlur(1.0))
        canv.save(dst_p)
    elif mode == "round_bottom_and_paws":
        scale = int(1024 * 0.68) / max(cw, ch)
        nw, nh = round(cw * scale), round(ch * scale)
        crop_r = crop.resize((nw, nh), Image.LANCZOS)
        canv = Image.new("RGB", (1024, 1024), (255, 255, 255))
        px = (1024 - nw) // 2
        py = (1024 - nh) // 2 - 30
        canv.paste(crop_r, (px, py))
        strip_h = 36
        bottom_strip = crop_r.crop((0, nh - strip_h, nw, nh))
        ext_h = 56
        bottom_ext = bottom_strip.resize((nw, ext_h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
        white_bg = Image.new("RGB", (nw, ext_h), (255, 255, 255))
        seeded = Image.blend(white_bg, bottom_ext, 0.65)
        canv.paste(seeded, (px, py + nh - 12))
        canv.save(dst_p)
    else:
        shutil.copyfile(src_cut, dst_p)


def clean_and_center(src: Path, dst: Path) -> None:
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
    E2E_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = E2E_DIR / "inputs"
    out_dir = E2E_DIR / "outputs"
    clean_dir = E2E_DIR / "clean"
    for d in (in_dir, out_dir, clean_dir):
        d.mkdir(exist_ok=True)

    NEG_BASE = (
        "photo, photorealistic, 3d render, airbrushed, glossy, shiny, plastic, "
        "heavy black outline, sticker, cross-hatching, pen hatching, "
        "branch, twig, rock, boulder, water, pond, water puddle, water ripple, reflection, water reflection, "
        "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
        "multiple animals, duplicate, tiling, text, watermark, border, cropped body, cut off, "
        "yellow spots on shell, box turtle, speckled background, confetti"
    )
    STYLE_CLAUSE = (
        "flat stylized gouache wildlife illustration, crisp distinct painted color steps, "
        "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
    )

    vlm_records = {}
    prompts = []
    for i, (slug, common, cut_p, wiki) in enumerate(TEST_CASES):
        res = call_gemma27b(cut_p, common, wiki)
        vlm_records[slug] = res
        print(f"\n[{slug}] ({res['_sec']}s):\n{json.dumps(res, indent=2)}", flush=True)

        inp_p = in_dir / f"{i:02d}_{slug}.png"
        prepare_seeded_input(cut_p, inp_p, res.get("completion_mode", "none"))

        article = "an" if common[0].lower() in "aeiou" else "a"
        p_text = (
            f"{article} {common.lower()}, {res['pose_and_view']}, "
            f"{res['visible_colors_and_markings']}, {res['completed_anatomy_clause']}, "
            f"{STYLE_CLAUSE} --n {NEG_BASE} --d 42"
        )
        prompts.append(p_text)

    (E2E_DIR / "vlm_cutout_outputs.json").write_text(json.dumps(vlm_records, indent=2), encoding="utf-8")
    pfile = E2E_DIR / "prompts.txt"
    pfile.write_text("\n".join(prompts), encoding="utf-8")

    # CRITICAL: Free gemma3:27b (17GB) from Ollama VRAM before launching SDXL!
    requests.post(f"{OLLAMA}/api/generate", json={"model": "gemma3:27b", "keep_alive": 0}, timeout=15)
    time.sleep(2)

    subprocess.run([
        str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
        "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        "--outdir", str(out_dir),
        "--network_module", "networks.lora",
        "--network_weights", str(LORA / "output" / "animl_style_v4-step00000750.safetensors"),
        "--network_mul", "0.85",
        "--max_embeddings_multiples", "3",
        "--from_file", str(pfile),
        "--image_path", str(in_dir),
        "--use_original_file_name",
        "--strength", "0.68",
        "--W", "1024", "--H", "1024",
        "--steps", "28", "--sampler", "euler_a",
        "--scale", "6.5", "--bf16", "--sdpa",
        "--batch_size", "1", "--images_per_prompt", "1",
    ], check=True)

    for p in out_dir.glob("*.png"):
        clean_and_center(p, clean_dir / p.name)

    # Assemble comparison sheet:
    # 1. Raw Cutout | 2. gemma3:27b Seeded Input | 3. Old v4 (No VLM) | 4. New gemma3:27b + Seeded Output
    try:
        font = ImageFont.truetype("arialbd.ttf", 21)
        font_sm = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    cols = [
        "1. Original Cutout",
        "2. gemma3:27b Seeded Input",
        "3. Old v4 (Short Prompt)",
        "4. gemma3:27b Prompt + Seed",
    ]
    cw, ch = 360, 360
    W = len(cols) * cw
    H = 48 + len(TEST_CASES) * (ch + 32)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    for ci, t in enumerate(cols):
        d.text((ci * cw + 14, 12), t, fill=(20, 20, 20), font=font)

    old_map = {
        "mule_deer": "04_mule_deer.png",
        "pond_slider": "08_pond_slider.png",
        "green-winged_teal": "03_green-winged_teal.png",
        "american_pika": "07_american_pika.png",
        "california_ground_squirrel": "06_california_ground_squirrel.png",
    }

    for ri, (slug, common, cut_p, _) in enumerate(TEST_CASES):
        mode = vlm_records[slug].get("completion_mode", "none")
        y_top = 48 + ri * (ch + 32)
        d.rectangle([0, y_top, W, y_top + 32], fill=(232, 232, 236))
        d.text((14, y_top + 6), f"{common} ({slug}) — gemma3:27b Cutout Inspection (completion_mode: {mode})", fill=(10, 10, 10), font=font_sm)
        files = [
            cut_p,
            in_dir / f"{ri:02d}_{slug}.png",
            WORK / "clean_str_0.76" / old_map[slug],
            clean_dir / f"{ri:02d}_{slug}.png",
        ]
        for ci, fp in enumerate(files):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            if fp.exists():
                im = Image.open(fp).convert("RGB")
                im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
                cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 32))
            d.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "gemma27b_cutout_e2e_compare.jpg", quality=92)
    print("Saved gemma27b_cutout_e2e_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
