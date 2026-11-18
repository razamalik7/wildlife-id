"""Unified Safe Pipeline Test across All 8 Common Watersnakes:
- Uses the EXACT same full checkpoint (`animl_style_v6-step00001250.safetensors`, U-Net + Text Encoder intact)
  and the EXACT same STYLE_V6 / NEG_V6 prompt structure as our Mammals, Birds, Lizards, and Turtles in `evaluate_v6_suite.py`.
- Implements 225-token Multi-Chunk SDXL Prompt Encoding (`encode_prompt_sdxl_multichunk`, matching `--max_embeddings_multiples 3`
  in `sdxl_gen_img.py`) so ZERO tokens are ever truncated by CLIP's 77-token window.
- Uses 100% automated `Depth-Anything-V2` + Bilateral Canny Dual ControlNet with ZERO per-image manual masks.
- Automatically mirrors left-facing snakes during SDXL inference (and mirrors back after) so the head aligns with the LoRA's
  right-facing canonical orientation without hallucinating heads on right-hand coil bends.
"""

import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
PREV_WORK = Path(__file__).parent / "batch_run_controlnet_wiki_all8"
DUAL_WORK = Path(__file__).parent / "batch_run_dual_cnet_all8"
WORK = Path(__file__).parent / "batch_run_unified_safe_all8"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

SNAKE_8_CANDS = [
    ("00_opt1_222925393", "222925393", "Option 1 (222925393): S-Coil on Rock", False),
    ("01_opt2_428226899", "428226899", "Option 2 (428226899): Side-Profile S-Loop", False),
    ("02_opt3_13608867",  "13608867",  "Option 3 (13608867): Layered Resting Coil", False),
    ("03_opt4_284103518", "284103518", "Option 4 (284103518): Flat S-Curve on Ground", False),
    ("04_opt5_20386464",  "20386464",  "Option 5 (20386464): S-Coil on Flat Stone", True),
    ("05_opt6_8478653",   "8478653",   "Option 6 (8478653): Tight Spiral Coil", True),
    ("06_opt7_526919366", "526919366", "Option 7 (526919366): Basking S-Coil", False),
    ("07_opt8_13810420",  "13810420",  "Option 8 (13810420): Raised-Head S-Curve", True),
]

# Exact 100% parity with evaluate_v6_suite.py (Mammals, Birds, Turtles, Lizards)
STYLE_V6 = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink fur and contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_V6_UNIFIED = (
    "two heads, double headed, multiple heads, extra head, head on tail end, "
    "weird bulges, lumpy swollen body, unrealistic shape, non-smooth coils, "
    "impossible geometry, disconnected or fused coils, uneven pinched tapering, warped deformed anatomy, "
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy shell, airbrushed, glossy, shiny, specular highlight, plastic, "
    "watercolor, gouache, painterly brush swirls, oil painting, blurry fur, stark white fur streaks, harsh contrast, "
    "heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two tails, extra limbs, extra feet, extra paws, row of feet, claws on belly, fused legs, tiling, text, watermark, border, cropped body, cut off"
)


def encode_prompt_sdxl_multichunk(
    pipe: StableDiffusionXLControlNetImg2ImgPipeline,
    prompt: str,
    negative_prompt: str,
    max_chunks: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encodes SDXL prompts across up to `max_chunks` (3 * 75 = 225 tokens) without ANY 77-token truncation!
    Matches `--max_embeddings_multiples 3` in sd-scripts.
    """
    device = pipe._execution_device
    dtype = pipe.unet.dtype

    def _encode_single(text: str, tokenizer, text_encoder, is_te2: bool = False):
        # Tokenize without truncation
        raw_ids = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        if pad is None:
            pad = eos

        # Determine number of 75-token chunks needed (1..max_chunks)
        n_chunks = max(1, min(max_chunks, (len(raw_ids) + 74) // 75))
        hidden_chunks = []
        pooled_first = None

        for c in range(n_chunks):
            seg = raw_ids[c * 75 : (c + 1) * 75]
            if len(seg) < 75:
                seg = seg + [pad] * (75 - len(seg))
            chunk_ids = torch.tensor([[bos] + seg + [eos]], dtype=torch.long, device=device)
            out = text_encoder(chunk_ids, output_hidden_states=True)
            # SDXL always uses penultimate hidden state (-2)
            hidden = out.hidden_states[-2]
            hidden_chunks.append(hidden)
            if is_te2 and c == 0:
                pooled_first = out[0]

        return torch.cat(hidden_chunks, dim=1), pooled_first, n_chunks

    # Determine required chunk count across pos and neg so both tensors have identical sequence length
    ids_pos = pipe.tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
    ids_neg = pipe.tokenizer(negative_prompt, truncation=False, add_special_tokens=False).input_ids
    req_chunks = max(1, min(max_chunks, (max(len(ids_pos), len(ids_neg)) + 74) // 75))

    def _encode_pair(text: str, n_chunks: int):
        raw1 = pipe.tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        raw2 = pipe.tokenizer_2(text, truncation=False, add_special_tokens=False).input_ids
        bos1, eos1 = pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id
        bos2, eos2 = pipe.tokenizer_2.bos_token_id, pipe.tokenizer_2.eos_token_id
        pad1 = pipe.tokenizer.pad_token_id if pipe.tokenizer.pad_token_id is not None else eos1
        pad2 = pipe.tokenizer_2.pad_token_id if pipe.tokenizer_2.pad_token_id is not None else 0

        h1_list, h2_list = [], []
        pooled = None
        for c in range(n_chunks):
            s1 = raw1[c * 75 : (c + 1) * 75]
            s2 = raw2[c * 75 : (c + 1) * 75]
            s1 = s1 + [pad1] * (75 - len(s1))
            s2 = s2 + [pad2] * (75 - len(s2))
            t1 = torch.tensor([[bos1] + s1 + [eos1]], dtype=torch.long, device=device)
            t2 = torch.tensor([[bos2] + s2 + [eos2]], dtype=torch.long, device=device)
            o1 = pipe.text_encoder(t1, output_hidden_states=True)
            o2 = pipe.text_encoder_2(t2, output_hidden_states=True)
            h1_list.append(o1.hidden_states[-2])
            h2_list.append(o2.hidden_states[-2])
            if c == 0:
                pooled = o2[0]
        h1 = torch.cat(h1_list, dim=1)
        h2 = torch.cat(h2_list, dim=1)
        prompt_embeds = torch.cat([h1, h2], dim=-1).to(dtype=dtype)
        return prompt_embeds, pooled.to(dtype=dtype)

    with torch.no_grad():
        pos_embeds, pos_pooled = _encode_pair(prompt, req_chunks)
        neg_embeds, neg_pooled = _encode_pair(negative_prompt, req_chunks)
    return pos_embeds, neg_embeds, pos_pooled, neg_pooled


def strip_head_eye_clauses(raw_traits: str) -> str:
    """Keep all body color/pattern clauses from Gemma3:27b + Wiki, while removing explicit 'head/eyes/pupils'
    clauses so SDXL cross-attention does not hallucinate extra eyes or heads on curved body coils."""
    clauses = [c.strip() for c in raw_traits.split(",") if c.strip()]
    kept = []
    for c in clauses:
        low = c.lower()
        if any(w in low for w in ("pupil", "eye", "head", "labial")):
            continue
        kept.append(c)
    return ", ".join(kept)


def build_automated_conditions(in_dir: Path, depth_dir: Path, canny_dir: Path) -> None:
    """100% automated Depth-Anything-V2 + Bilateral Canny conditioning (ZERO manual per-image masks)."""
    in_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    canny_dir.mkdir(parents=True, exist_ok=True)

    if all((depth_dir / f"{s}.png").exists() and (canny_dir / f"{s}.png").exists() for s, _, _, _ in SNAKE_8_CANDS):
        print("Automated Depth-Anything-V2 + Bilateral Canny conditions already cached.", flush=True)
        return

    # Copy the shadow-cleaned cutouts from DUAL_WORK / inputs_clean
    for stem, _, _, _ in SNAKE_8_CANDS:
        shutil.copyfile(DUAL_WORK / "inputs_clean" / f"{stem}.png", in_dir / f"{stem}.png")

    print("Running Depth-Anything-V2-Small-hf (unclipped min-max normalization)...", flush=True)
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to("cuda")

    for stem, _, _, _ in SNAKE_8_CANDS:
        im = Image.open(in_dir / f"{stem}.png").convert("RGB")
        arr = np.array(im)
        fg = np.any(arr < 245, axis=-1)
        fg_u8 = fg.astype(np.uint8) * 255
        fg_inner = cv2.erode(fg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask_inv = (~fg).astype(np.uint8) * 255

        inputs = processor(images=im, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = depth_model(**inputs)
            post = processor.post_process_depth_estimation(outputs, target_sizes=[(1024, 1024)])
        depth = post[0]["predicted_depth"].cpu().numpy()

        # True min-max normalization (ZERO 98th-percentile clipping so no highlight plateau ring ever forms!)
        d_fg = depth[fg]
        d_min, d_max = float(d_fg.min()), float(d_fg.max())
        d_norm = np.zeros_like(depth, dtype=np.uint8)
        d_scaled = (depth - d_min) / max(1e-5, d_max - d_min)
        d_norm[fg] = (45 + d_scaled[fg] * 210).astype(np.uint8)
        Image.fromarray(np.stack([d_norm] * 3, axis=-1)).save(depth_dir / f"{stem}.png")

        # 1. Pure 3D coil-over-coil occlusion jumps from smooth unclipped depth
        depth_inp = cv2.inpaint(d_norm, mask_inv, 5, cv2.INPAINT_TELEA)
        depth_smooth = cv2.GaussianBlur(depth_inp, (9, 9), 2.0)
        depth_edges = cv2.Canny(depth_smooth, 24, 62)
        depth_edges = cv2.bitwise_and(depth_edges, fg_inner)
        depth_edges_strong = cv2.dilate(depth_edges, np.ones((2, 2), np.uint8), iterations=1)

        # 2. Bilateral-smoothed RGB Canny with automatic contrast-adaptive threshold
        #    so low-contrast gray snakes (like Option 8) still retain internal crossband edges on body bends!
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray_inp = cv2.inpaint(gray, mask_inv, 5, cv2.INPAINT_TELEA)
        sm = cv2.bilateralFilter(gray_inp, d=9, sigmaColor=62, sigmaSpace=62)
        sm = cv2.bilateralFilter(sm, d=9, sigmaColor=62, sigmaSpace=62)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
        sm_eq = cv2.bilateralFilter(clahe.apply(sm), d=7, sigmaColor=45, sigmaSpace=45)

        t_low, t_high = 48, 120
        for cand_low, cand_high in [(48, 120), (34, 88), (22, 58)]:
            rgb_edges = cv2.Canny(sm_eq, cand_low, cand_high)
            rgb_edges = cv2.bitwise_and(rgb_edges, fg_inner)
            density = float((rgb_edges[fg_inner > 0] > 0).mean())
            if density >= 0.052:
                break

        # Prune tiny isolated dirt specks (< 18px) automatically
        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(rgb_edges)
        rgb_clean = np.zeros_like(rgb_edges)
        for i in range(1, num_l):
            if stats[i, cv2.CC_STAT_AREA] >= 14 or max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 14:
                rgb_clean[labels == i] = 255

        sil_edges = cv2.Canny(fg_u8, 50, 150)
        combined = cv2.bitwise_or(depth_edges_strong, cv2.bitwise_or(rgb_clean, sil_edges))
        Image.fromarray(np.stack([combined] * 3, axis=-1)).save(canny_dir / f"{stem}.png")

    del depth_model
    del processor
    torch.cuda.empty_cache()


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
    build_automated_conditions(in_dir, depth_dir, canny_dir)

    vlm_traits = json.loads((PREV_WORK / "vlm_watersnake_8.json").read_text(encoding="utf-8"))

    print("Loading Dual ControlNet + animl_style_v6_unet_only.safetensors...", flush=True)
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
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        controlnet=MultiControlNetModel([cnet_canny, cnet_depth]),
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6_unet_only.safetensors")
    pipe.fuse_lora(lora_scale=1.05)
    pipe.to("cuda")

    raw_d = WORK / "raw_s060"
    clean_d = WORK / "clean_s060"
    raw_d.mkdir(parents=True, exist_ok=True)
    clean_d.mkdir(parents=True, exist_ok=True)

    for idx, (stem, obs_id, title, do_mirror) in enumerate(SNAKE_8_CANDS):
        out_raw = raw_d / f"{stem}.png"
        out_clean = clean_d / f"{stem}.png"
        if out_clean.exists():
            shutil.copyfile(out_clean, ARTIFACT_DIR / f"unified_safe_opt{idx+1}.png")
            continue

        init_im = Image.open(in_dir / f"{stem}.png").convert("RGB")
        canny_im = Image.open(canny_dir / f"{stem}.png").convert("RGB")
        depth_im = Image.open(depth_dir / f"{stem}.png").convert("RGB")

        # Canonical right-facing orientation during SDXL inference when needed
        if do_mirror:
            init_im = ImageOps.mirror(init_im)
            canny_im = ImageOps.mirror(canny_im)
            depth_im = ImageOps.mirror(depth_im)

        # Full, untruncated prompt matching evaluate_v6_suite.py format!
        body_traits = strip_head_eye_clauses(vlm_traits[stem])
        prompt = (
            f"animl_gouache_style, field guide illustration of a Common watersnake, "
            f"{body_traits}, {STYLE_V6}"
        )

        pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_sdxl_multichunk(
            pipe, prompt, NEG_V6_UNIFIED, max_chunks=3
        )
        print(f"[Unified Safe {idx+1}/8] {stem} (mirror={do_mirror}, embeds={tuple(pos_embeds.shape)})...", flush=True)

        gen = torch.Generator(device="cuda").manual_seed(42)
        res = pipe(
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            image=init_im,
            control_image=[canny_im, depth_im],
            strength=0.60,
            controlnet_conditioning_scale=[0.66, 0.36],
            control_guidance_start=0.0,
            control_guidance_end=[0.84, 0.68],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
        ).images[0]

        if do_mirror:
            res = ImageOps.mirror(res)
        res.save(out_raw)
        if out_clean.exists():
            out_clean.unlink()
        clean_post(res, out_clean)
        shutil.copyfile(out_clean, ARTIFACT_DIR / f"unified_safe_opt{idx+1}.png")

    # Build comparison sheet comparing Previous Best vs Unified Safe (Full Checkpoint + 225-token Multi-Chunk + Automated CNet)
    col_w = 360
    row_h = 390
    header_h = 56
    cols = 4
    rows = len(SNAKE_8_CANDS)
    sheet = Image.new("RGB", (col_w * cols, header_h + row_h * rows), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    try:
        f_hdr = ImageFont.truetype("arialbd.ttf", 16)
        f_sub = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        f_hdr = ImageFont.load_default()
        f_sub = f_hdr

    headers = [
        "1. Cleaned Photo Cutout",
        "2. Automated Depth-Anything-V2",
        "3. Automated Hybrid Canny",
        "4. Unified Full-v6 + 225-Token Output",
    ]
    for c_idx, h_text in enumerate(headers):
        draw.text((c_idx * col_w + 12, 16), h_text, fill=(20, 20, 20), font=f_hdr)

    for r_idx, (stem, obs_id, title, facing) in enumerate(SNAKE_8_CANDS):
        y0 = header_h + r_idx * row_h
        draw.line([(0, y0), (col_w * cols, y0)], fill=(210, 210, 210), width=2)
        imgs = [
            Image.open(in_dir / f"{stem}.png").convert("RGB"),
            Image.open(depth_dir / f"{stem}.png").convert("RGB"),
            Image.open(canny_dir / f"{stem}.png").convert("RGB"),
            Image.open(clean_d / f"{stem}.png").convert("RGB"),
        ]
        for c_idx, im in enumerate(imgs):
            x0 = c_idx * col_w
            thumb = im.resize((col_w - 16, col_w - 16), Image.LANCZOS)
            sheet.paste(thumb, (x0 + 8, y0 + 26))
        draw.text((12, y0 + 5), f"{title} [facing={facing}]", fill=(10, 50, 110), font=f_sub)

    sheet_out = ARTIFACT_DIR / "v6_unified_safe_all8_compare.jpg"
    sheet.save(sheet_out, quality=94)
    print(f"Saved {sheet_out}!", flush=True)


if __name__ == "__main__":
    main()
