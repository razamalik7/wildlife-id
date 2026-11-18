"""Automated Pipeline Test v2 for All 8 Common Watersnakes:
1. Dual ControlNet (`diffusers/controlnet-canny-sdxl-1.0-small` + `diffusers/controlnet-depth-sdxl-1.0-small`):
   - `Depth-Anything-V2-Small-hf` generates a 3D cylindrical depth map of every coil so SDXL knows the exact 3D volume and front-to-back coil ordering (eliminating coil-blending & tube warping on dark snakes).
   - Hybrid 3D-Depth-Occlusion + Bilateral RGB Canny map locks the exact single-line head/eye/jaw outline, coil-over-coil occlusion boundaries, and dorsal crossbands without scale clutter.
2. Zero-Truncation 77-Token Prompt Ordering:
   - Places the core flat cel-shaded gouache + dark charcoal-umber outline tokens inside tokens 1..42 followed by concise Wikipedia + Gemma3:27b color/marking traits so 0 tokens are truncated by CLIP.
3. De-poisoned `animl_style_v6_unet_only.safetensors` (`lora_scale=1.05`).
"""

import json
import re
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
ALL8_DIR = Path(__file__).parent / "batch_run_all8_snakes"
PREV_WORK = Path(__file__).parent / "batch_run_controlnet_wiki_all8"
WORK = Path(__file__).parent / "batch_run_dual_cnet_all8"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SNAKE_8_CANDS = [
    ("00_opt1_222925393", "222925393", "Option 1 (222925393): S-Coil on Rock"),
    ("01_opt2_428226899", "428226899", "Option 2 (428226899): Side-Profile S-Loop"),
    ("02_opt3_13608867",  "13608867",  "Option 3 (13608867): Layered Resting Coil"),
    ("03_opt4_284103518", "284103518", "Option 4 (284103518): Flat S-Curve on Ground"),
    ("04_opt5_20386464",  "20386464",  "Option 5 (20386464): S-Coil on Flat Stone"),
    ("05_opt6_8478653",   "8478653",   "Option 6 (8478653): Tight Spiral Coil"),
    ("06_opt7_526919366", "526919366", "Option 7 (526919366): Basking S-Coil"),
    ("07_opt8_13810420",  "13810420",  "Option 8 (13810420): Raised-Head S-Curve"),
]

NEG_PROMPT = (
    "two heads, double headed, multiple heads, extra head, head on tail end, "
    "weird bulges, lumpy swollen body, unrealistic shape, non-smooth coils, "
    "impossible geometry, disconnected or fused coils, uneven pinched tapering, "
    "warped deformed anatomy, claws, legs, limbs, toes, open mouth, tongue, "
    "photo, photorealistic, 3d render, glossy, shiny, specular highlight, stark white stripes, white rings"
)


def compact_traits(raw_traits: str, max_words: int = 24) -> str:
    """Keep the most informative color/pattern words from Gemma3:27b + Wiki so the full prompt stays <= 75 CLIP tokens."""
    t = re.sub(r"\b(snake|watersnake|nerodia|sipedon|body|with|having|featuring|showing|visible)\b", "", raw_traits, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(" ,.;")
    return t


def prepare_inputs_and_conditions(in_dir: Path, depth_dir: Path, canny_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    canny_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare clean cutouts
    for stem, _, _ in SNAKE_8_CANDS:
        src = ALL8_DIR / "inputs" / f"{stem}.png"
        dst = in_dir / f"{stem}.png"
        arr = np.array(Image.open(src).convert("RGB"))

        if stem == "03_opt4_284103518":
            # Smooth shed-skin/twig bump on upper-left outer edge (y=390..495)
            for y in range(390, 495):
                t = (y - 390) / 105.0
                x_lim = int(round(216 - 84 * (t ** 0.82)))
                arr[y, :x_lim, :] = 255
        elif stem == "06_opt7_526919366":
            # Remove dark log shadow under head coil
            for x in range(620, 910):
                if x < 765:
                    t = (765 - x) / 145.0
                    y_lim = int(round(662 - 34 * (t ** 1.6)))
                else:
                    t = (x - 765) / 128.0
                    y_lim = int(round(662 - 68 * (t ** 2.0)))
                y_lim = max(560, min(800, y_lim))
                arr[y_lim:, x, :] = 255
            arr[680:, 800:, :] = 255
        elif stem == "07_opt8_13810420":
            # Remove right-edge gravel shadow cleanly across full bend (y=395..595) with zero spikes
            for y in range(395, 595):
                if y < 495:
                    t = (495 - y) / 100.0
                    x_lim = int(round(846 - 68.0 * (t ** 1.8)))
                else:
                    t = (y - 495) / 100.0
                    x_lim = int(round(846 - 118.0 * (t ** 1.65)))
                arr[y, x_lim:, :] = 255

        Image.fromarray(arr).save(dst)

    # 2. Run Depth-Anything-V2-Small-hf on all 8 clean cutouts
    print("Running Depth-Anything-V2-Small-hf on all 8 cutouts...", flush=True)
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to("cuda")

    for stem, _, _ in SNAKE_8_CANDS:
        im = Image.open(in_dir / f"{stem}.png").convert("RGB")
        arr = np.array(im)
        fg = np.any(arr < 245, axis=-1)
        inputs = processor(images=im, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = depth_model(**inputs)
            post = processor.post_process_depth_estimation(outputs, target_sizes=[(1024, 1024)])
        depth = post[0]["predicted_depth"].cpu().numpy()
        d_fg = depth[fg]
        d_min, d_max = np.percentile(d_fg, 2), np.percentile(d_fg, 98)
        d_norm = np.zeros_like(depth, dtype=np.uint8)
        d_scaled = np.clip((depth - d_min) / max(1e-5, d_max - d_min), 0, 1)
        d_norm[fg] = (45 + d_scaled[fg] * 210).astype(np.uint8)
        # Smooth 1px jaggies inside depth map
        Image.fromarray(np.stack([d_norm] * 3, axis=-1)).save(depth_dir / f"{stem}.png")

        # 3. Build Hybrid 3D-Depth-Occlusion + Bilateral RGB Canny map
        fg_u8 = fg.astype(np.uint8) * 255
        fg_inner = cv2.erode(fg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask_inv = (~fg).astype(np.uint8) * 255
        depth_inp = cv2.inpaint(d_norm, mask_inv, 5, cv2.INPAINT_TELEA)

        # Pure 3D coil-over-coil occlusion edges from Depth-Anything-V2
        depth_edges = cv2.Canny(depth_inp, 16, 42)
        depth_edges = cv2.bitwise_and(depth_edges, fg_inner)
        depth_edges_strong = cv2.dilate(depth_edges, np.ones((2, 2), np.uint8), iterations=1)

        # Bilateral-smoothed RGB Canny for head/eye/jaw and major dorsal crossbands
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray_inp = cv2.inpaint(gray, mask_inv, 5, cv2.INPAINT_TELEA)
        smooth = cv2.bilateralFilter(gray_inp, d=9, sigmaColor=65, sigmaSpace=65)
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=65, sigmaSpace=65)
        rgb_edges = cv2.Canny(smooth, 55, 135)
        rgb_edges = cv2.bitwise_and(rgb_edges, fg_inner)

        # Clean 1px outer silhouette
        sil_edges = cv2.Canny(fg_u8, 50, 150)

        combined = cv2.bitwise_or(depth_edges_strong, rgb_edges)
        combined = cv2.bitwise_or(combined, sil_edges)
        Image.fromarray(np.stack([combined] * 3, axis=-1)).save(canny_dir / f"{stem}.png")

    del depth_model
    del processor
    torch.cuda.empty_cache()
    print("Unloaded Depth-Anything-V2 from VRAM.", flush=True)


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


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    in_dir = WORK / "inputs"
    depth_dir = WORK / "depth"
    canny_dir = WORK / "canny"
    prepare_inputs_and_conditions(in_dir, depth_dir, canny_dir)

    vlm_cache_p = PREV_WORK / "vlm_watersnake_8.json"
    vlm_traits = json.loads(vlm_cache_p.read_text(encoding="utf-8"))

    print("Loading Dual SDXL ControlNet (Canny + Depth) + animl_style_v6_unet_only.safetensors...", flush=True)
    cnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0-small",
        torch_dtype=torch.bfloat16,
        variant="fp16",
    )
    cnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small",
        torch_dtype=torch.bfloat16,
        variant="fp16",
    )
    multi_cnet = MultiControlNetModel([cnet_canny, cnet_depth])

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        controlnet=multi_cnet,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6_unet_only.safetensors")
    pipe.fuse_lora(lora_scale=1.05)
    pipe.to("cuda")

    configs = [
        ("dual_s060", 0.60, [0.52, 0.48], 6.5, 0.82),
        ("dual_s066", 0.66, [0.58, 0.52], 6.5, 0.84),
    ]

    for tag, strength, cnet_scales, cfg, c_end in configs:
        raw_d = WORK / f"raw_{tag}"
        clean_d = WORK / f"clean_{tag}"
        raw_d.mkdir(parents=True, exist_ok=True)
        clean_d.mkdir(parents=True, exist_ok=True)

        for idx, (stem, obs_id, title) in enumerate(SNAKE_8_CANDS):
            out_raw = raw_d / f"{stem}.png"
            out_clean = clean_d / f"{stem}.png"
            if not out_raw.exists():
                init_im = Image.open(in_dir / f"{stem}.png").convert("RGB")
                canny_im = Image.open(canny_dir / f"{stem}.png").convert("RGB")
                depth_im = Image.open(depth_dir / f"{stem}.png").convert("RGB")

                short_traits = compact_traits(vlm_traits[stem], max_words=22)
                # Style clause FIRST so 0 tokens are truncated by CLIP's 77-token window!
                prompt = (
                    f"animl_gouache_style, field guide illustration of a Common watersnake, "
                    f"flat cel-shaded color zones with stepped shading, clean thin dark charcoal-umber outline, "
                    f"muted matte natural colors, {short_traits}, isolated on white background"
                )
                print(f"[{tag} {idx+1}/8] {stem} (s={strength}, scales={cnet_scales})...", flush=True)
                gen = torch.Generator(device="cuda").manual_seed(42)
                res = pipe(
                    prompt=prompt,
                    negative_prompt=NEG_PROMPT,
                    image=init_im,
                    control_image=[canny_im, depth_im],
                    strength=strength,
                    controlnet_conditioning_scale=cnet_scales,
                    control_guidance_start=0.0,
                    control_guidance_end=c_end,
                    guidance_scale=cfg,
                    num_inference_steps=38,
                    generator=gen,
                ).images[0]
                res.save(out_raw)
            clean_post(Image.open(out_raw), out_clean)

    # Build 5-column comparison sheet: Cutout | 3D Depth | Hybrid Canny | Dual CNet s=0.60 | Dual CNet s=0.66
    print("Building comparison sheet...", flush=True)
    col_w = 340
    row_h = 370
    header_h = 56
    cols = 5
    rows = len(SNAKE_8_CANDS)
    sheet = Image.new("RGB", (col_w * cols, header_h + row_h * rows), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    try:
        f_hdr = ImageFont.truetype("arialbd.ttf", 17)
        f_sub = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        f_hdr = ImageFont.load_default()
        f_sub = f_hdr

    headers = [
        "1. Cleaned Photo Cutout",
        "2. Depth-Anything-V2 (3D Coils)",
        "3. Hybrid Depth+RGB Canny",
        "4. Dual CNet (Depth+Canny) s=0.60",
        "5. Dual CNet (Depth+Canny) s=0.66",
    ]
    for c_idx, h_text in enumerate(headers):
        draw.text((c_idx * col_w + 12, 16), h_text, fill=(20, 20, 20), font=f_hdr)

    for r_idx, (stem, obs_id, title) in enumerate(SNAKE_8_CANDS):
        y0 = header_h + r_idx * row_h
        draw.line([(0, y0), (col_w * cols, y0)], fill=(210, 210, 210), width=2)
        imgs = [
            Image.open(in_dir / f"{stem}.png").convert("RGB"),
            Image.open(depth_dir / f"{stem}.png").convert("RGB"),
            Image.open(canny_dir / f"{stem}.png").convert("RGB"),
            Image.open(WORK / "clean_dual_s060" / f"{stem}.png").convert("RGB"),
            Image.open(WORK / "clean_dual_s066" / f"{stem}.png").convert("RGB"),
        ]
        for c_idx, im in enumerate(imgs):
            x0 = c_idx * col_w
            thumb = im.resize((col_w - 16, col_w - 16), Image.LANCZOS)
            sheet.paste(thumb, (x0 + 8, y0 + 26))
        draw.text((12, y0 + 5), f"{title}", fill=(10, 50, 110), font=f_sub)

    sheet_out = ARTIFACT_DIR / "v6_dual_controlnet_depth_canny_all8_compare.jpg"
    sheet.save(sheet_out, quality=93)
    print(f"Saved comparison sheet to {sheet_out}", flush=True)


if __name__ == "__main__":
    main()
