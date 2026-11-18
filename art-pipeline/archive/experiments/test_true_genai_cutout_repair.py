"""
Compare True Generative AI Photo-Cutout Repair & Anatomy Completion BEFORE Stylization
on:
- 02_black_tailed_jackrabbit (missing grass-occluded paws/legs + grass in mouth)
- 08_red_winged_blackbird (diagonal reed/twig & leaf clump across body/legs)
- 10_pied_billed_grebe (swimming waterbird cut off at waterline -> complete belly + webbed/lobed feet)

Compares:
1. Original BiRefNet Cutout & Previous Stylized Output
2. Local 9-Channel SDXL Inpainting (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` at strength=0.98) + Stylized Output
3. Instruction-Guided Multimodal Image Edit (Gemini Flash Image) Cutout + Stylized Output
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
    StableDiffusionXLInpaintPipeline,
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

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
GENAI_WORK = Path(__file__).parent / "batch_run_true_genai_repair"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

TEST_3 = [
    ("02_black_tailed_jackrabbit", "Black-tailed Jackrabbit", "Lepus californicus", "Mammalia", "mammal"),
    ("08_red_winged_blackbird", "Red-winged Blackbird", "Agelaius phoeniceus", "Aves", "bird"),
    ("10_pied_billed_grebe", "Pied-billed Grebe", "Podilymbus podiceps", "Aves", "duck"),
]


def normalize_edited_cutout_to_1024(im_path: Path) -> Image.Image:
    """Ensure the GenAI-edited photo cutout is 1024x1024 on pure #FFFFFF white background."""
    im = Image.open(im_path).convert("RGB")
    if im.size != (1024, 1024):
        im = im.resize((1024, 1024), Image.Resampling.LANCZOS)
    # Run clean_post (BiRefNet) to ensure background is 100% pure #FFFFFF and subject is cleanly isolated
    tmp_out = GENAI_WORK / f"_tmp_{im_path.name}"
    if tmp_out.exists():
        tmp_out.unlink()
    clean_post(im, tmp_out)
    rgba = Image.open(tmp_out).convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3] > 20
    ys, xs = np.where(alpha)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = rgba.crop((x0, y0, x1 + 1, y1 + 1))
        cw, ch = crop.size
        scale = min(780.0 / max(cw, ch), 1.35)
        nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        crop_r = crop.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canvas.alpha_composite(crop_r, ((1024 - nw) // 2, (1024 - nh) // 2))
        return canvas.convert("RGB")
    return im


def run_local_sdxl_inpaint_cutout(
    inpaint_pipe: StableDiffusionXLInpaintPipeline,
    stem: str,
    orig_cut: Image.Image,
) -> Image.Image:
    """Run local SDXL Inpainting (`StableDiffusionXLInpaintPipeline`) at strength=0.96."""
    mask = np.zeros((1024, 1024), dtype=np.uint8)

    if stem == "02_black_tailed_jackrabbit":
        # Mask the grass in mouth/chest AND the bottom leg/paw zone where grass occluded the feet
        cv2.rectangle(mask, (280, 500), (370, 710), 255, -1)
        cv2.rectangle(mask, (320, 715), (785, 905), 255, -1)
        prompt = (
            "a realistic wildlife photograph of a black-tailed jackrabbit (Lepus californicus) "
            "standing with four complete furry tan-buff legs and full paws visible on a solid pure white background, "
            "clean fur, sharp focus, high resolution wildlife photography"
        )
        neg = "grass, green stems, missing feet, cut off legs, floating, deformed paws, blurry, shadow"
    elif stem == "08_red_winged_blackbird":
        # Mask the diagonal twig across the wing/belly and the leaf/reed area around the legs
        poly = np.array([[340, 480], [565, 360], [670, 405], [560, 745], [340, 745]], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        prompt = (
            "a realistic wildlife photograph of a male red-winged blackbird (Agelaius phoeniceus) "
            "with glossy black feathers and two clean black bird legs and feet with toes visible, "
            "isolated on a solid pure white background, no branches or twigs"
        )
        neg = "branch, twig, reed, stick, dead leaf, perch, brown stick, blurry, missing legs"
    else:  # 10_pied_billed_grebe
        # Shift swimming grebe slightly up so there is room below the waterline for complete belly & webbed/lobed feet
        shifted = Image.new("RGB", (1024, 1024), (255, 255, 255))
        shifted.paste(orig_cut.crop((0, 90, 1024, 1024)), (0, 0))
        orig_cut = shifted
        # Mask the flat waterline cut at y=600..890, x=230..790
        cv2.rectangle(mask, (220, 585), (795, 895), 255, -1)
        prompt = (
            "a realistic wildlife photograph of a full-body pied-billed grebe (Podilymbus podiceps) standing, "
            "showing its complete rounded brown-and-buff belly and two dark gray-green grebe legs and lobed webbed feet "
            "isolated on a solid pure white background, sharp focus"
        )
        neg = "water, ripples, cut off, flat bottom, swimming, reflection, blurry, missing feet"

    mask_blur = cv2.GaussianBlur(mask, (25, 25), 8.0)
    mask_pil = Image.fromarray(mask_blur)
    gen = torch.Generator(device="cuda").manual_seed(42)
    out = inpaint_pipe(
        prompt=prompt,
        negative_prompt=neg,
        image=orig_cut,
        mask_image=mask_pil,
        strength=0.96,
        guidance_scale=7.5,
        num_inference_steps=32,
        generator=gen,
    ).images[0]

    # Blend smoothly using feathered mask so unmasked regions remain 100% untouched
    alpha_f = (mask_blur.astype(np.float32) / 255.0)[..., None]
    comp = (np.array(orig_cut) * (1.0 - alpha_f) + np.array(out) * alpha_f).astype(np.uint8)
    tmp_p = GENAI_WORK / f"_inpaint_raw_{stem}.png"
    Image.fromarray(comp).save(tmp_p)
    return normalize_edited_cutout_to_1024(tmp_p)


def main() -> None:
    GENAI_WORK.mkdir(parents=True, exist_ok=True)
    meta = json.loads((WORK / "fresh_20_metadata.json").read_text(encoding="utf-8"))

    print("Loading SDXL + Canny ControlNet...", flush=True)
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
    pipe.to("cuda")

    inpaint_pipe = StableDiffusionXLInpaintPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )

    # Step 1: Run Local SDXL Inpainting on the 3 cutouts
    for stem, common, sci, cls_name, subtype in TEST_3:
        out_inpaint_cut = GENAI_WORK / f"sdxl_inpaint_cutout_{stem}.png"
        orig_cut = Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB")
        rep_inpaint = run_local_sdxl_inpaint_cutout(inpaint_pipe, stem, orig_cut)
        rep_inpaint.save(out_inpaint_cut)
        print(f"[SDXL Local Inpaint] Saved {out_inpaint_cut.name}", flush=True)

    # Step 2: Normalize the Multimodal GenAI (Nano Banana / Gemini Flash Image) edited cutouts
    for stem, common, sci, cls_name, subtype in TEST_3:
        matches = sorted(ARTIFACT_DIR.glob(f"genai_cutout_{stem}*"))
        dst_genai = GENAI_WORK / f"multimodal_cutout_{stem}.png"
        if matches:
            src_genai = matches[-1]
            norm_im = normalize_edited_cutout_to_1024(src_genai)
            norm_im.save(dst_genai)
            shutil.copyfile(dst_genai, ARTIFACT_DIR / f"genai_cutout_{stem}_norm.png")
            print(f"[Multimodal GenAI Cutout] Normalized {src_genai.name} -> {dst_genai.name}", flush=True)

    # Step 3: Load animl_style_v6 and stylize both sets of repaired cutouts!
    print("Loading animl_style_v6-step00001250.safetensors...", flush=True)
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6-step00001250.safetensors")
    pipe.fuse_lora(lora_scale=1.0)

    for stem, common, sci, cls_name, subtype in TEST_3:
        traits = meta[stem].get("visual_traits") or meta[stem].get("wiki_clauses") or f"natural {common} coloration"
        prompt = (
            f"animl_gouache_style, a {common}, full body wildlife illustration, "
            f"{traits}, {STYLE_V6}"
        )
        pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_sdxl_multichunk(
            pipe, prompt, NEG_V6_UNIFIED, max_chunks=3
        )

        for tag, cut_path in [
            ("sdxl_inpaint", GENAI_WORK / f"sdxl_inpaint_cutout_{stem}.png"),
            ("multimodal", GENAI_WORK / f"multimodal_cutout_{stem}.png"),
        ]:
            if not cut_path.exists():
                continue
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

            out_clean = GENAI_WORK / f"stylized_{tag}_{stem}.png"
            if out_clean.exists():
                out_clean.unlink()
            clean_post(res, out_clean)
            shutil.copyfile(out_clean, ARTIFACT_DIR / f"stylized_{tag}_{stem}.png")
            print(f"[Stylized {tag}] Saved stylized_{tag}_{stem}.png", flush=True)

    # Build 5-column comparison sheet:
    # 1. Original Flawed Cutout | 2. Local SDXL Inpaint Cutout | 3. Local SDXL Inpaint + v6 Style | 4. Multimodal GenAI Cutout | 5. Multimodal GenAI + v6 Style
    col_w = 310
    row_h = 350
    header_h = 54
    cols = 5
    rows = len(TEST_3)
    sheet = Image.new("RGB", (col_w * cols, header_h + row_h * rows), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    try:
        f_hdr = ImageFont.truetype("arialbd.ttf", 14)
        f_sub = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        f_hdr = ImageFont.load_default()
        f_sub = f_hdr

    headers = [
        "1. Original Flawed Cutout",
        "2. SDXL Inpaint Cutout (Local)",
        "3. SDXL Inpaint + v6 Stylized",
        "4. Multimodal GenAI Cutout",
        "5. Multimodal GenAI + v6 Stylized",
    ]
    for c_idx, h_text in enumerate(headers):
        draw.text((c_idx * col_w + 10, 16), h_text, fill=(20, 20, 20), font=f_hdr)

    for r_idx, (stem, common, sci, cls_name, subtype) in enumerate(TEST_3):
        y0 = header_h + r_idx * row_h
        draw.line([(0, y0), (col_w * cols, y0)], fill=(200, 200, 200), width=1)
        draw.text((10, y0 + 6), f"{stem[:2]}. {common} ({sci}) [{cls_name}/{subtype}]", fill=(15, 50, 100), font=f_sub)

        paths = [
            WORK / "cutouts_1024" / f"{stem}.png",
            GENAI_WORK / f"sdxl_inpaint_cutout_{stem}.png",
            GENAI_WORK / f"stylized_sdxl_inpaint_{stem}.png",
            GENAI_WORK / f"multimodal_cutout_{stem}.png",
            GENAI_WORK / f"stylized_multimodal_{stem}.png",
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
            thumb = im.resize((290, 290), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (c_idx * col_w + 10, y0 + 34))

    out_sheet = ARTIFACT_DIR / "v6_true_genai_repair_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}!", flush=True)


if __name__ == "__main__":
    main()
