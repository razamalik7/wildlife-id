"""A/B Comparison on the 18 Non-Snake Species from Fresh-20 (USING EXACT SAME CUTOUT INPUTS):
Compares:
  Col 1: Normalized BiRefNet Cutout (same input as before)
  Col 2: Previous Run (With Depth + Canny ControlNet)
  Col 3: Without Depth (Canny-Only ControlNet, depth scale = 0.0)
  Col 4: Pure SDXL Img2Img (No Depth & No Canny ControlNet, s=0.62 — evaluate_v6_suite.py mode)
"""

import json
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
from test_fresh_20_e2e import (
    FRESH_20_SPECIES,
    STYLE_V6,
    NEG_V6_UNIFIED,
    encode_prompt_sdxl_multichunk,
    clean_post,
)

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")


def main() -> None:
    cutout_dir = WORK / "cutouts_1024"
    canny_dir = WORK / "canny"
    depth_dir = WORK / "depth"
    prev_clean_dir = WORK / "clean_out"

    nodepth_raw = WORK / "raw_nodepth_canny_only"
    nodepth_clean = WORK / "clean_nodepth_canny_only"
    pure_raw = WORK / "raw_pure_img2img"
    pure_clean = WORK / "clean_pure_img2img"
    for d in (nodepth_raw, nodepth_clean, pure_raw, pure_clean):
        d.mkdir(parents=True, exist_ok=True)

    meta = json.loads((WORK / "fresh_20_metadata.json").read_text(encoding="utf-8"))
    non_snakes = [s for s in FRESH_20_SPECIES if s[5] != "snake"]

    print("Loading SDXL + ControlNet + FULL animl_style_v6-step00001250.safetensors...", flush=True)
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
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6-step00001250.safetensors")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.to("cuda")

    for idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(non_snakes):
        init_im = Image.open(cutout_dir / f"{stem}.png").convert("RGB")
        canny_im = Image.open(canny_dir / f"{stem}.png").convert("RGB")
        depth_im = Image.open(depth_dir / f"{stem}.png").convert("RGB")

        traits = meta[stem].get("visual_traits") or meta[stem].get("wiki_clauses") or f"natural {common} coloration"
        prompt = (
            f"animl_gouache_style, a {common}, full body wildlife illustration, "
            f"{traits}, {STYLE_V6}"
        )
        pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_sdxl_multichunk(
            pipe, prompt, NEG_V6_UNIFIED, max_chunks=3
        )

        # Mode A: Without Depth (Canny-Only ControlNet, depth scale = 0.0)
        out_nd_raw = nodepth_raw / f"{stem}.png"
        out_nd_clean = nodepth_clean / f"{stem}.png"
        if not out_nd_clean.exists():
            if subtype == "amphibian":
                strength, c_scales, c_ends = 0.55, [0.44, 0.0], [0.74, 1.0]
            else:
                strength, c_scales, c_ends = 0.58, [0.42, 0.0], [0.72, 1.0]
            print(f"[{idx+1}/18 No-Depth Canny-Only] {stem}...", flush=True)
            gen = torch.Generator(device="cuda").manual_seed(42)
            res = pipe(
                prompt_embeds=pos_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pos_pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                image=init_im,
                control_image=[canny_im, depth_im],
                strength=strength,
                controlnet_conditioning_scale=c_scales,
                control_guidance_start=0.0,
                control_guidance_end=c_ends,
                guidance_scale=6.0,
                num_inference_steps=36,
                generator=gen,
            ).images[0]
            res.save(out_nd_raw)
            clean_post(res, out_nd_clean)

        # Mode B: Pure SDXL Img2Img (Zero Depth & Zero Canny ControlNet — evaluate_v6_suite.py mode)
        out_p_raw = pure_raw / f"{stem}.png"
        out_p_clean = pure_clean / f"{stem}.png"
        if not out_p_clean.exists():
            strength_pure = 0.58 if subtype == "amphibian" else 0.62
            print(f"[{idx+1}/18 Pure Img2Img (No CNet)] {stem} (s={strength_pure})...", flush=True)
            gen = torch.Generator(device="cuda").manual_seed(42)
            res = pipe(
                prompt_embeds=pos_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pos_pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                image=init_im,
                control_image=[canny_im, depth_im],
                strength=strength_pure,
                controlnet_conditioning_scale=[0.0, 0.0],
                control_guidance_start=0.0,
                control_guidance_end=[1.0, 1.0],
                guidance_scale=6.0,
                num_inference_steps=36,
                generator=gen,
            ).images[0]
            res.save(out_p_raw)
            clean_post(res, out_p_clean)

    # Build 2 comparison sheets (9 species each) with 4 columns:
    # 1. Cutout Input | 2. With Depth + Canny | 3. Without Depth (Canny Only) | 4. Pure Img2Img (No Depth / No Canny)
    try:
        f_hdr = ImageFont.truetype("arialbd.ttf", 15)
        f_sub = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        f_hdr = ImageFont.load_default()
        f_sub = f_hdr

    for part_idx, chunk in enumerate([non_snakes[:9], non_snakes[9:]], start=1):
        col_w = 340
        row_h = 370
        header_h = 54
        cols = 4
        rows = len(chunk)
        sheet = Image.new("RGB", (col_w * cols, header_h + row_h * rows), (246, 246, 246))
        draw = ImageDraw.Draw(sheet)

        headers = [
            "1. Normalized BiRefNet Cutout",
            "2. With Depth + Canny CNet",
            "3. Without Depth (Canny Only)",
            "4. Pure Img2Img (No Depth, No Canny)",
        ]
        for c_idx, h_text in enumerate(headers):
            draw.text((c_idx * col_w + 10, 16), h_text, fill=(20, 20, 20), font=f_hdr)

        for r_idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(chunk):
            y0 = header_h + r_idx * row_h
            draw.line([(0, y0), (col_w * cols, y0)], fill=(210, 210, 210), width=2)
            imgs = [
                Image.open(cutout_dir / f"{stem}.png").convert("RGB"),
                Image.open(prev_clean_dir / f"{stem}.png").convert("RGB"),
                Image.open(nodepth_clean / f"{stem}.png").convert("RGB"),
                Image.open(pure_clean / f"{stem}.png").convert("RGB"),
            ]
            for c_idx, im in enumerate(imgs):
                x0 = c_idx * col_w
                thumb = im.resize((col_w - 16, col_w - 16), Image.LANCZOS)
                sheet.paste(thumb, (x0 + 8, y0 + 26))
            label = f"{stem[:2]}. {common} ({sci}) [{cls_name}/{subtype}]"
            draw.text((12, y0 + 5), label, fill=(10, 50, 110), font=f_sub)

        sheet_out = ARTIFACT_DIR / f"v6_fresh20_ab_depth_part{part_idx}.jpg"
        sheet.save(sheet_out, quality=93)
        print(f"Saved {sheet_out}", flush=True)


if __name__ == "__main__":
    main()
