"""Automated Pipeline Test for All 8 Common Watersnakes:
1. Variant-Aware Wikipedia Description + Gemma3:27b VLM prompt generation (`call_gemma27b`) for each photo cutout.
2. De-poisoned `animl_style_v6_unet_only.safetensors` (100% of all 722 U-Net style layers, zero poisoned `common_watersnake` Text Encoder tokens).
3. SDXL ControlNet (`diffusers/controlnet-canny-sdxl-1.0-small` via `StableDiffusionXLControlNetImg2ImgPipeline`)
   using a structural Canny edge map (locking internal coil overlaps, head/eye/jaw position, and band boundaries
   while smoothing out tiny individual scale grain so the interior paints in flat cel-shaded gouache).
"""

import base64
import json
import re
import shutil
import subprocess
import time
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, EulerAncestralDiscreteScheduler
from u2net_matte import remove
from normalize import subject_bbox

OLLAMA = "http://localhost:11434"
LORA = Path(__file__).parent / "lora"
ALL8_DIR = Path(__file__).parent / "batch_run_all8_snakes"
WORK = Path(__file__).parent / "batch_run_controlnet_wiki_all8"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SNAKE_8_CANDS = [
    ("00_opt1_222925393", "222925393", "Option 1 (222925393): S-Coil on Rock"),
    ("01_opt2_428226899", "428226899", "Option 2 (428226899): Side-Profile S-Loop"),
    ("02_opt3_13608867",  "13608867",  "Option 3 (13608867): Layered Resting Coil"),
    ("03_opt4_284103518", "284103518", "Option 4 (284103518): Flat S-Curve on Ground"),
    ("04_opt5_20386464",  "20386464",  "Option 5 (20386464): S-Coil on Flat Stone"),
    ("05_opt6_8478653",   "8478653",   "Option 6 (8478653): Tight Spiral Coil"),
    ("06_opt7_526919366", "526919366", "Option 7 (526919366): Basking S-Coil (Shadow Removed)"),
    ("07_opt8_13810420",  "13810420",  "Option 8 (13810420): Raised-Head S-Curve (Shadow Removed)"),
]

STYLE_CLAUSE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_PROMPT = (
    "two heads, double headed, multiple heads, extra head, head on tail end, "
    "weird bulges, lumpy swollen body, unrealistic shape, non-smooth coils, "
    "impossible geometry, disconnected or fused coils, uneven pinched tapering, "
    "warped deformed anatomy, claws, legs, limbs, toes, open mouth, tongue, "
    "photo, photorealistic, 3d render, glossy, shiny, specular highlight, stark white stripes, white rings"
)

PROMPT_TEMPLATE = """You are a wildlife illustrator inspecting a cutout image of a {common_name} alongside its Wikipedia Description.

Wikipedia Description for {common_name}:
\"\"\"{wiki_text}\"\"\"

Your job is to return ONLY one field in JSON: `visible_color_and_markings`.
- CRITICAL VARIANT RULE: Look closely at the attached image first to determine the actual base color and pattern of this individual {common_name} (e.g. warm terracotta-orange and dark brown crossbands vs. dark olive-brown with tan crossbands vs. pale gray-tan with brown dorsal blotches), and combine those true image colors with the species' diagnostic visible markings from Wikipedia (such as dark labial bars on the jaw, dorsal crossbands on the neck alternating with blotches posteriorly, round pupil).
- Write a 16-24 word comma-separated list of `[color/pattern] [body part]` phrases for ONLY the parts of the animal visible from this angle.
- DO NOT mention hidden underside parts (no 'ventral scales').
- DO NOT describe pose or camera angle.
- DO NOT mention environment, water, grass, rocks, or logs.
- DO NOT use the word 'pinkish' or 'white rings'.

Return STRICT JSON only:
{{
  "visible_color_and_markings": "16-24 word comma-separated list of visible `[color/pattern] [body part]` phrases matching the individual in the image"
}}"""


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


def call_gemma27b(cut_p: Path, common_name: str, wiki_text: str) -> str:
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
    return data.get("visible_color_and_markings", "").strip()


def make_structural_canny(rgb_im: Image.Image) -> Image.Image:
    """Create a structural Canny edge map that preserves:
    1. Outer silhouette edge
    2. Internal coil overlap boundaries (which tube crosses over which)
    3. Exact head outline, eye, and labial bar placement
    4. Major dorsal band contours
    while smoothing out tiny 2-3px photorealistic scale grain so ControlNet
    allows flat cel-shaded gouache painting inside the coils.
    """
    arr = np.array(rgb_im.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    fg = (np.any(arr < 245, axis=-1)).astype(np.uint8) * 255

    # Two-pass bilateral filter preserves coil boundaries & head/eye edges while flattening scale micro-texture
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=65, sigmaSpace=65)
    smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=65, sigmaSpace=65)

    edges_internal = cv2.Canny(smooth, 55, 135)
    # Also ensure strong outer silhouette contour + strong internal dark valley coil separators
    edges_sil = cv2.Canny(fg, 50, 150)
    combined = cv2.bitwise_or(edges_internal, edges_sil)

    # Dilate slightly (1px) so 128x128 latent downsample sees continuous coil boundaries
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=1)
    return Image.fromarray(np.stack([combined] * 3, axis=-1))


def clean_post(im: Image.Image, dst_path: Path) -> Image.Image:
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    cut = remove(im.convert("RGBA"))
    arr = np.array(cut)
    arr[arr[:, :, 3] < 165, 3] = 0
    cut = Image.fromarray(arr)
    box = subject_bbox(cut)
    if box is None:
        rgb = im.convert("RGB")
        rgb.save(dst_path)
        return rgb
    subj = cut.crop(box)
    scale = int(1024 * 0.78) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    rgb = canvas.convert("RGB")
    rgb.save(dst_path)
    return rgb


def prepare_inputs(in_dir: Path, canny_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    canny_dir.mkdir(parents=True, exist_ok=True)
    for stem, _, _ in SNAKE_8_CANDS:
        src = ALL8_DIR / "inputs" / f"{stem}.png"
        dst = in_dir / f"{stem}.png"
        if stem == "06_opt7_526919366":
            im7 = Image.open(src).convert("RGB")
            arr7 = np.array(im7)
            for x in range(620, 910):
                if x < 765:
                    t = (765 - x) / 145.0
                    y_lim = int(round(662 - 34 * (t ** 1.6)))
                else:
                    t = (x - 765) / 128.0
                    y_lim = int(round(662 - 68 * (t ** 2.0)))
                y_lim = max(560, min(800, y_lim))
                arr7[y_lim:, x, :] = 255
            Image.fromarray(arr7).save(dst)
        elif stem == "07_opt8_13810420":
            im8 = Image.open(src).convert("RGB")
            arr8 = np.array(im8)
            for y in range(440, 580):
                t = (y - 495) / 55.0
                x_lim = int(round(862 - 14.0 * (t ** 2)))
                arr8[y, x_lim:, :] = 255
            Image.fromarray(arr8).save(dst)
        else:
            shutil.copyfile(src, dst)

        im = Image.open(dst).convert("RGB")
        canny_im = make_structural_canny(im)
        canny_im.save(canny_dir / f"{stem}.png")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    in_dir = WORK / "inputs"
    canny_dir = WORK / "canny"
    prepare_inputs(in_dir, canny_dir)

    # 1. Automated Wikipedia + Gemma3:27b VLM descriptions for all 8 cutouts
    vlm_cache_p = WORK / "vlm_watersnake_8.json"
    vlm_traits: dict[str, str] = {}
    if vlm_cache_p.exists():
        vlm_traits = json.loads(vlm_cache_p.read_text(encoding="utf-8"))

    used_ollama = False
    wiki_txt = None
    for stem, _, _ in SNAKE_8_CANDS:
        if stem not in vlm_traits or not vlm_traits[stem]:
            if wiki_txt is None:
                wiki_txt = fetch_full_wiki_description("Common watersnake")
            used_ollama = True
            traits = call_gemma27b(in_dir / f"{stem}.png", "Common Watersnake", wiki_txt)
            vlm_traits[stem] = traits
            vlm_cache_p.write_text(json.dumps(vlm_traits, indent=2), encoding="utf-8")
            print(f"  [Gemma3:27b + Wiki] {stem}: {traits}", flush=True)
        else:
            print(f"  [Cached Gemma3:27b + Wiki] {stem}: {vlm_traits[stem]}", flush=True)

    if used_ollama:
        print("Unloading gemma3:27b from Ollama VRAM...", flush=True)
        requests.post(f"{OLLAMA}/api/generate", json={"model": "gemma3:27b", "keep_alive": 0}, timeout=15)
        time.sleep(2)

    # 2. Load SDXL ControlNet + de-poisoned v6_unet_only LoRA
    print("Loading SDXL ControlNet Canny + animl_style_v6_unet_only.safetensors...", flush=True)
    cnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0-small",
        torch_dtype=torch.bfloat16,
        variant="fp16",
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        controlnet=cnet,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6_unet_only.safetensors")
    pipe.fuse_lora(lora_scale=1.05)
    pipe.to("cuda")

    configs = [
        ("cnet_s056", 0.56, 0.65, 6.0),
        ("cnet_s064", 0.64, 0.72, 6.5),
        ("cnet_s072", 0.72, 0.78, 6.5),
    ]

    for tag, strength, cnet_scale, cfg in configs:
        raw_d = WORK / f"raw_{tag}"
        clean_d = WORK / f"clean_{tag}"
        raw_d.mkdir(parents=True, exist_ok=True)
        clean_d.mkdir(parents=True, exist_ok=True)
        for stem, _, _ in SNAKE_8_CANDS:
            out_clean = clean_d / f"{stem}.png"
            if out_clean.exists():
                continue
            im_in = Image.open(in_dir / f"{stem}.png").convert("RGB")
            im_cnet = Image.open(canny_dir / f"{stem}.png").convert("RGB")
            traits = vlm_traits[stem]
            prompt = (
                f"animl_gouache_style, a common watersnake, full body wildlife illustration, "
                f"single animal, one head only, {traits}, {STYLE_CLAUSE}"
            )
            gen = torch.Generator(device="cuda").manual_seed(42)
            out_im = pipe(
                prompt=prompt,
                negative_prompt=NEG_PROMPT,
                image=im_in,
                control_image=im_cnet,
                strength=strength,
                controlnet_conditioning_scale=cnet_scale,
                control_guidance_end=0.88,
                guidance_scale=cfg,
                num_inference_steps=30,
                generator=gen,
            ).images[0]
            out_im.save(raw_d / f"{stem}.png")
            clean_post(out_im, out_clean)
            print(f"  Generated {tag} / {stem}.png", flush=True)

    # 3. Build 5-column comparison sheet across all 8 snakes
    CELL = 340
    HEADER_H = 50
    ROW_LABEL_H = 44
    cols = [
        ("1. Cutout Input (Shadow-Free)", in_dir),
        ("2. Structural Canny Map", canny_dir),
        ("3. ControlNet + Wiki (s=0.56)", WORK / "clean_cnet_s056"),
        ("4. ControlNet + Wiki (s=0.64)", WORK / "clean_cnet_s064"),
        ("5. ControlNet + Wiki (s=0.72)", WORK / "clean_cnet_s072"),
    ]
    W = len(cols) * CELL
    H = HEADER_H + len(SNAKE_8_CANDS) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 15)
        font_r = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    for c_idx, (title, _) in enumerate(cols):
        draw.text((c_idx * CELL + 10, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, pid, lbl) in enumerate(SNAKE_8_CANDS):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        traits_short = vlm_traits.get(stem, "")[:135]
        draw.text((12, y0 + 4), lbl, fill=(15, 20, 60), font=font_b)
        draw.text((12, y0 + 23), f"Gemma3:27b + Wiki: {traits_short}", fill=(40, 40, 50), font=font_r)
        for c_idx, (_, dpath) in enumerate(cols):
            p = dpath / f"{stem}.png"
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", (CELL - 8, CELL - 8), (255, 255, 255))
                bg.paste(im, ((CELL - 8 - im.width) // 2, (CELL - 8 - im.height) // 2))
                sheet.paste(bg, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_controlnet_wiki_gemma_all8_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
