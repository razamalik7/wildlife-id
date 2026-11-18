"""
Stylize all 7 repaired photo cutouts from the Unified GenAI Cutout-Repair Instruction Set:
- 02_black_tailed_jackrabbit
- 06_house_finch
- 08_red_winged_blackbird
- 09_california_scrub_jay
- 10_pied_billed_grebe (tucked/paddling swimming posture, NOT standing!)
- 14_southern_alligator_lizard
- 15_spiny_softshell
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

sys.path.insert(0, str(Path(__file__).parent))
from test_fresh_20_e2e import (
    STYLE_V6,
    NEG_V6_UNIFIED,
    clean_post,
    encode_prompt_sdxl_multichunk,
)
from test_pre_stylization_crop_repair import build_canny_from_repaired
from test_true_genai_cutout_repair import normalize_edited_cutout_to_1024

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
GENAI_WORK = Path(__file__).parent / "batch_run_true_genai_repair"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

ALL_7 = [
    ("02_black_tailed_jackrabbit", "Black-tailed Jackrabbit", "Lepus californicus", "Mammalia", "mammal", "genai_cutout_02_black_tailed_jackrabbit"),
    ("06_house_finch", "House Finch", "Haemorhous mexicanus", "Aves", "bird", "genai_perch_06_house_finch"),
    ("08_red_winged_blackbird", "Red-winged Blackbird", "Agelaius phoeniceus", "Aves", "bird", "genai_perch_08_red_winged_blackbird"),
    ("09_california_scrub_jay", "California Scrub-Jay", "Aphelocoma californica", "Aves", "bird", "genai_unified_09_california_scrub_jay"),
    ("10_pied_billed_grebe", "Pied-billed Grebe", "Podilymbus podiceps", "Aves", "duck", "genai_unified_10_pied_billed_grebe"),
    ("14_southern_alligator_lizard", "Southern Alligator Lizard", "Elgaria multicarinata", "Reptilia", "lizard", "REGION_LOCKED"),
    ("15_spiny_softshell", "Spiny Softshell", "Apalone spinifera", "Reptilia", "turtle", "REGION_LOCKED"),
]


def normalize_white_bg_cutout_preserving_perch(im_path: Path) -> Image.Image:
    """Normalize a GenAI-repaired cutout on #FFFFFF without letting BiRefNet strip the short perch stub!"""
    im = Image.open(im_path).convert("RGB")
    if im.size != (1024, 1024):
        im = im.resize((1024, 1024), Image.Resampling.LANCZOS)
    arr = np.array(im)
    fg = np.any(arr < 245, axis=-1).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.where(fg > 0)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        rgba = np.dstack([arr, fg])
        crop = Image.fromarray(rgba, "RGBA").crop((x0, y0, x1 + 1, y1 + 1))
        cw, ch = crop.size
        scale = min(780.0 / max(cw, ch), 1.35)
        nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        crop_r = crop.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canvas.alpha_composite(crop_r, ((1024 - nw) // 2, (1024 - nh) // 2))
        return canvas.convert("RGB")
    return im


def main() -> None:
    GENAI_WORK.mkdir(parents=True, exist_ok=True)
    meta = json.loads((WORK / "fresh_20_metadata.json").read_text(encoding="utf-8"))
    prev_repair_dir = Path(__file__).parent / "batch_run_pre_stylization_repair"

    # Normalize all 7 GenAI edited cutouts (using region-locked repair for 14 & 15 so head/hands/scales never drift!)
    for stem, common, sci, cls_name, subtype, prefix in ALL_7:
        dst_genai = GENAI_WORK / f"unified_cutout_{stem}.png"
        if prefix == "REGION_LOCKED":
            # Use the region-locked cutout & stylized output where 100% of the head, hands, and body scales are preserved!
            shutil.copyfile(prev_repair_dir / "cutouts_repaired" / f"{stem}.png", dst_genai)
            shutil.copyfile(prev_repair_dir / "clean_out" / f"{stem}.png", GENAI_WORK / f"stylized_unified_{stem}.png")
            shutil.copyfile(prev_repair_dir / "clean_out" / f"{stem}.png", ARTIFACT_DIR / f"stylized_unified_{stem}.png")
            continue
        matches = sorted(ARTIFACT_DIR.glob(f"{prefix}*"))
        if matches:
            src_genai = matches[-1]
            if stem in ("06_house_finch", "08_red_winged_blackbird"):
                norm_im = normalize_white_bg_cutout_preserving_perch(src_genai)
            else:
                norm_im = normalize_edited_cutout_to_1024(src_genai)
            norm_im.save(dst_genai)
            print(f"[Unified GenAI Cutout] Normalized {src_genai.name} -> {dst_genai.name}", flush=True)

    print("Loading SDXL + Canny ControlNet + animl_style_v6...", flush=True)
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

    for stem, common, sci, cls_name, subtype, prefix in ALL_7:
        cut_path = GENAI_WORK / f"unified_cutout_{stem}.png"
        out_clean = GENAI_WORK / f"stylized_unified_{stem}.png"
        if not cut_path.exists():
            continue
        # Reuse already-stylized 02, 09, 10, 14, 15 to save time
        if stem == "02_black_tailed_jackrabbit":
            prev_multimodal = GENAI_WORK / f"stylized_multimodal_{stem}.png"
            if prev_multimodal.exists():
                shutil.copyfile(prev_multimodal, out_clean)
                shutil.copyfile(out_clean, ARTIFACT_DIR / f"stylized_unified_{stem}.png")
                continue
        if stem not in ("06_house_finch", "08_red_winged_blackbird") and out_clean.exists():
            continue

        traits = meta[stem].get("visual_traits") or meta[stem].get("wiki_clauses") or f"natural {common} coloration"
        prompt = (
            f"animl_gouache_style, a {common}, full body wildlife illustration, "
            f"{traits}, {STYLE_V6}"
        )
        pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_sdxl_multichunk(
            pipe, prompt, NEG_V6_UNIFIED, max_chunks=3
        )

        cut_im = Image.open(cut_path).convert("RGB")
        canny_im = build_canny_from_repaired(cut_im)
        gen = torch.Generator(device="cuda").manual_seed(42)
        res = pipe(
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            image=cut_im,
            control_image=[canny_im, canny_im],
            strength=0.58,
            controlnet_conditioning_scale=[0.44, 0.0],
            control_guidance_start=0.0,
            control_guidance_end=[0.74, 1.0],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
        ).images[0]

        if out_clean.exists():
            out_clean.unlink()
        clean_post(res, out_clean)
        # Preserve the short perch stub / repaired silhouette present in cut_im so BiRefNet doesn't strip the perch!
        clean_im = Image.open(out_clean).convert("RGBA")
        c_arr = np.array(clean_im)
        r_arr = np.array(res.convert("RGB"))
        rep_fg = np.any(np.array(cut_im) < 245, axis=-1).astype(np.uint8) * 255
        rep_fg = cv2.erode(rep_fg, np.ones((3, 3), np.uint8), iterations=1)
        rep_alpha = cv2.GaussianBlur(rep_fg.astype(np.float32), (5, 5), 1.0).astype(np.uint8)
        union_alpha = np.maximum(c_arr[:, :, 3], rep_alpha)
        c_arr[:, :, :3] = np.where(union_alpha[..., None] > 12, r_arr, 255)
        c_arr[:, :, 3] = union_alpha
        Image.fromarray(c_arr, "RGBA").save(out_clean)
        shutil.copyfile(out_clean, ARTIFACT_DIR / f"stylized_unified_{stem}.png")
        print(f"[Stylized Unified] Saved stylized_unified_{stem}.png", flush=True)

    # Build 4-column comparison sheet across all 7 species:
    # 1. Original Flawed Cutout | 2. Unified GenAI Repaired Cutout | 3. Previous Output (Before Repair) | 4. New Output (After GenAI Repair)
    col_w = 340
    row_h = 360
    header_h = 54
    cols = 4
    rows = len(ALL_7)
    sheet = Image.new("RGB", (col_w * cols, header_h + row_h * rows), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    try:
        f_hdr = ImageFont.truetype("arialbd.ttf", 15)
        f_sub = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        f_hdr = ImageFont.load_default()
        f_sub = f_hdr

    headers = [
        "1. Original BiRefNet Cutout",
        "2. GenAI Repaired Cutout (Pre-Style)",
        "3. Previous Output (Before Repair)",
        "4. New Output (After GenAI Repair)",
    ]
    for c_idx, h_text in enumerate(headers):
        draw.text((c_idx * col_w + 12, 16), h_text, fill=(20, 20, 20), font=f_hdr)

    for r_idx, (stem, common, sci, cls_name, subtype, _) in enumerate(ALL_7):
        y0 = header_h + r_idx * row_h
        draw.line([(0, y0), (col_w * cols, y0)], fill=(200, 200, 200), width=1)
        draw.text((12, y0 + 6), f"{stem[:2]}. {common} ({sci}) [{cls_name}/{subtype}]", fill=(15, 50, 100), font=f_sub)

        paths = [
            WORK / "cutouts_1024" / f"{stem}.png",
            GENAI_WORK / f"unified_cutout_{stem}.png",
            WORK / "clean_out" / f"{stem}.png",
            GENAI_WORK / f"stylized_unified_{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if not p.exists():
                continue
            im = Image.open(p)
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                im = bg
            else:
                im = im.convert("RGB")
            thumb = im.resize((310, 310), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (c_idx * col_w + 15, y0 + 34))

    out_sheet = ARTIFACT_DIR / "v6_unified_genai_all7_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}!", flush=True)


if __name__ == "__main__":
    main()
