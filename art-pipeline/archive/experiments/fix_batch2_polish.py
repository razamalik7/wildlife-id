"""
Final Polish Pass for Batch 2 (20 species):
1. `06_painted_bunting`, `07_cedar_waxwing`, `08_belted_kingfisher`:
   Ensure the short wooden perch stub under the bird's gripping feet is 100% solid and never faded to white by SDXL.
2. `19_horn_shark`:
   Sanitize 'Horn Shark' -> 'a brown spotted Heterodontus benthic shark (Heterodontus francisci)' (removing the literal token 'Horn')
   and smooth the diagonal aquarium-glass reflection streak on the snout before Canny + SDXL runs.
3. `10_hooded_merganser`, `16_northern_leopard_frog`, `17_spring_peeper`:
   Apply localized cel-shaded Photo-Iris Color Restoration so the male Hooded Merganser's bright golden-yellow eye
   and the frogs' golden-bronze iris rings match their reference photos.
4. Rebuild `v6_batch2_20_part1.jpg` and `v6_batch2_20_part2.jpg`.
"""
import gc
import json
import shutil
import sys
from pathlib import Path

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
from test_fresh_20_e2e import STYLE_V6, NEG_V6_UNIFIED, clean_post, encode_prompt_sdxl_multichunk
from test_fresh_20_batch2 import BATCH2_20_SPECIES, WORK, LORA, ARTIFACT_DIR
from test_pre_stylization_crop_repair import build_canny_from_repaired


def restore_cel_perch_stub(stem: str, y_min_perch: int) -> None:
    """Ensure the short wooden perch stub under the bird's gripping toes in `cutouts_final/{stem}.png`
    is 100% solid and cel-shaded in `clean_out/{stem}.png` even where SDXL faded part of the twig to white."""
    cut_im = Image.open(WORK / "cutouts_final" / f"{stem}.png").convert("RGB")
    raw_im = Image.open(WORK / "raw_stylized" / f"{stem}.png").convert("RGB")
    out_p = WORK / "clean_out" / f"{stem}.png"
    post_rgba = Image.open(out_p).convert("RGBA")

    cut_arr = np.array(cut_im)
    raw_arr = np.array(raw_im)
    post_arr = np.array(post_rgba)

    # Create a flat cel-shaded gouache version of the cutout's wooden twig
    cel_cut = cv2.bilateralFilter(cut_arr, d=11, sigmaColor=50, sigmaSpace=50)
    cel_cut = cv2.bilateralFilter(cel_cut, d=11, sigmaColor=50, sigmaSpace=50)

    fg_cut = np.any(cut_arr < 240, axis=-1)
    fg_cut[:y_min_perch, :] = False
    fg_eroded = cv2.erode(fg_cut.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0

    # Wherever the perch stub in `raw_arr` was faded toward white (mean RGB > 210) or stripped in `post_arr`,
    # fill with the cel-shaded cutout perch stub!
    faded_or_missing = fg_eroded & ((post_arr[:, :, 3] < 180) | (raw_arr.mean(axis=-1) > 205))
    post_arr[faded_or_missing, :3] = cel_cut[faded_or_missing]
    post_arr[fg_eroded, 3] = 255

    # Add a subtle charcoal-umber outline along the outer boundary of the restored perch stub
    contour_ring = cv2.dilate(fg_eroded.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
    outline_px = contour_ring & (~fg_eroded) & (post_arr[:, :, 3] < 128)
    post_arr[outline_px, :3] = (58, 48, 42)
    post_arr[outline_px, 3] = 255

    Image.fromarray(post_arr).save(out_p)
    shutil.copyfile(out_p, ARTIFACT_DIR / f"batch2_{stem}.png")


def restore_eye_iris(stem: str, eye_box: tuple[int, int, int, int], min_cut_l: float = 95.0) -> None:
    """Restore bright golden/light iris from `cutouts_final/{stem}.png` inside `eye_box`."""
    cut_arr = np.array(Image.open(WORK / "cutouts_final" / f"{stem}.png").convert("RGB"))
    out_p = WORK / "clean_out" / f"{stem}.png"
    styl_arr = np.array(Image.open(out_p).convert("RGBA"))

    x1, y1, x2, y2 = eye_box
    cut_crop = cut_arr[y1:y2, x1:x2]
    styl_crop = styl_arr[y1:y2, x1:x2, :3].copy()

    cut_smooth = cv2.bilateralFilter(cut_crop, d=9, sigmaColor=45, sigmaSpace=45)
    lab_s = cv2.cvtColor(cut_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    bright_px = lab_s[:, :, 0] > min_cut_l
    lab_s[bright_px, 0] = np.clip(lab_s[bright_px, 0] * 1.15 + 12.0, 0, 248)
    cut_smooth = cv2.cvtColor(lab_s.astype(np.uint8), cv2.COLOR_LAB2RGB)

    h, w = cut_crop.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((yy - h / 2.0) / (h / 2.0)) ** 2 + ((xx - w / 2.0) / (w / 2.0)) ** 2)

    cut_l = cv2.cvtColor(cut_crop, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    styl_l = cv2.cvtColor(styl_crop, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)

    ellipse_w = np.clip((0.92 - dist) / 0.22, 0.0, 1.0)
    iris_mask = ((cut_l > min_cut_l) & (styl_l < 215.0)).astype(np.float32)
    iris_mask = cv2.GaussianBlur(iris_mask * ellipse_w, (5, 5), 1.0)[:, :, None]

    styl_arr[y1:y2, x1:x2, :3] = np.clip(
        styl_crop.astype(np.float32) * (1.0 - 0.90 * iris_mask) + cut_smooth.astype(np.float32) * (0.90 * iris_mask),
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(styl_arr).save(out_p)
    shutil.copyfile(out_p, ARTIFACT_DIR / f"batch2_{stem}.png")


def fix_horn_shark() -> None:
    """Smooth the diagonal glass-reflection streak on `19_horn_shark` and re-render without the literal word 'Horn'."""
    stem = "19_horn_shark"
    cut_orig = Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB")
    arr = np.array(cut_orig)

    # In `19_horn_shark`, x=680..850, y=420..560 has a bright pale diagonal glass reflection across the brown head.
    # Let's smooth and clamp abnormally bright reflection pixels inside the head region to the surrounding warm brown skin tone!
    fg = np.any(arr < 242, axis=-1)
    head_box = np.zeros_like(fg)
    head_box[420:555, 660:860] = True
    roi_mask = fg & head_box
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Clamp glare (L > 148) in the head reflection zone down toward the median head skin tone (L ~ 125, a ~ 134, b ~ 146)
    glare = roi_mask & (lab[:, :, 0] > 142.0)
    lab[glare, 0] = 124.0 + (lab[glare, 0] - 142.0) * 0.25
    lab[glare, 1] = 134.0
    lab[glare, 2] = 146.0
    arr_fixed = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    # Smooth the repaired glare patch slightly
    arr_blur = cv2.bilateralFilter(arr_fixed, d=11, sigmaColor=45, sigmaSpace=45)
    arr[glare] = arr_blur[glare]

    cut_im = Image.fromarray(arr)
    cut_im.save(WORK / "cutouts_final" / f"{stem}.png")
    canny_im = build_canny_from_repaired(cut_im)
    canny_im.save(WORK / "canny_final" / f"{stem}.png")
    depth_im = Image.open(WORK / "depth_final" / f"{stem}.png").convert("RGB")

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

    prompt = (
        f"animl_gouache_style, a brown spotted Heterodontus benthic shark (Heterodontus francisci), "
        f"full body wildlife illustration, warm tan-brown body covered in small dark brown-black spots, "
        f"smooth blunt rounded snout, two dorsal fins, broad pectoral fins, {STYLE_V6}"
    )
    neg_prompt = "horns, spikes on nose, whiskers, barbels, glass reflection, diagonal lines, tiger stripes, " + NEG_V6_UNIFIED
    pos_e, neg_e, pos_p, neg_p = encode_prompt_sdxl_multichunk(pipe, prompt, neg_prompt, max_chunks=3)
    gen = torch.Generator(device="cuda").manual_seed(42)
    res_im = pipe(
        prompt_embeds=pos_e,
        negative_prompt_embeds=neg_e,
        pooled_prompt_embeds=pos_p,
        negative_pooled_prompt_embeds=neg_p,
        image=cut_im,
        control_image=[canny_im, depth_im],
        strength=0.54,
        controlnet_conditioning_scale=[0.46, 0.25],
        control_guidance_start=0.0,
        control_guidance_end=[0.75, 0.60],
        guidance_scale=6.0,
        num_inference_steps=36,
        generator=gen,
    ).images[0]
    res_im.save(WORK / "raw_stylized" / f"{stem}.png")

    del pipe, cnet_canny, cnet_depth
    gc.collect()
    torch.cuda.empty_cache()

    out_p = WORK / "clean_out" / f"{stem}.png"
    if out_p.exists():
        out_p.unlink()
    clean_post(res_im, out_p)
    shutil.copyfile(out_p, ARTIFACT_DIR / f"batch2_{stem}.png")


def rebuild_sheets() -> None:
    for part_idx, slice_sp in [(1, BATCH2_20_SPECIES[:10]), (2, BATCH2_20_SPECIES[10:20])]:
        col_w, row_h, header_h = 340, 340, 56
        w = 4 * col_w
        h = header_h + len(slice_sp) * row_h
        sheet = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        try:
            font_hdr = ImageFont.truetype("arialbd.ttf", 22)
            font_lbl = ImageFont.truetype("arialbd.ttf", 17)
        except Exception:
            font_hdr = ImageFont.load_default()
            font_lbl = ImageFont.load_default()

        headers = [
            "1. Selected Ref Photo",
            "2. Pre-Stylization Cutout",
            "3. Canny Structure Map",
            "4. Final Stylized (v6)",
        ]
        draw.rectangle([0, 0, w, header_h], fill=(35, 38, 45))
        for c_i, h_text in enumerate(headers):
            draw.text((c_i * col_w + 16, 16), h_text, fill=(255, 255, 255), font=font_hdr)

        for r_i, (stem, common, sci, _, _, _) in enumerate(slice_sp):
            y0 = header_h + r_i * row_h
            p_photo = WORK / "photos_selected" / f"{stem}.png"
            p_cut = WORK / "cutouts_final" / f"{stem}.png"
            p_canny = WORK / "canny_final" / f"{stem}.png"
            p_styl = WORK / "clean_out" / f"{stem}.png"
            for c_i, p in enumerate([p_photo, p_cut, p_canny, p_styl]):
                x0 = c_i * col_w
                if p.exists():
                    im = Image.open(p)
                    if im.mode == "RGBA":
                        bg = Image.new("RGB", im.size, (255, 255, 255))
                        bg.paste(im, mask=im.split()[3])
                        im = bg
                    else:
                        im = im.convert("RGB")
                    sheet.paste(im.resize((col_w - 16, row_h - 42), Image.Resampling.LANCZOS), (x0 + 8, y0 + 34))
                draw.rectangle([x0, y0, x0 + col_w, y0 + row_h], outline=(220, 220, 220), width=1)
            draw.text((12, y0 + 8), f"{stem}: {common} ({sci})", fill=(20, 20, 20), font=font_lbl)

        out_sheet = ARTIFACT_DIR / f"v6_batch2_20_part{part_idx}.jpg"
        sheet.save(out_sheet, quality=92)
        print(f"Updated Batch 2 Part {part_idx} sheet: {out_sheet}", flush=True)


def main() -> None:
    # 1. Ensure solid short wooden perch stubs on perched birds
    restore_cel_perch_stub("06_painted_bunting", y_min_perch=660)
    restore_cel_perch_stub("07_cedar_waxwing", y_min_perch=610)
    restore_cel_perch_stub("08_belted_kingfisher", y_min_perch=580)

    # 2. Restore golden/light irises on 10_hooded_merganser, 16_northern_leopard_frog, 17_spring_peeper
    restore_eye_iris("10_hooded_merganser", (718, 402, 762, 444), min_cut_l=110.0)
    restore_eye_iris("16_northern_leopard_frog", (808, 336, 878, 396), min_cut_l=95.0)
    restore_eye_iris("17_spring_peeper", (778, 394, 852, 456), min_cut_l=95.0)

    # 3. Fix 19_horn_shark glass reflection + 'Horn' token sanitization
    fix_horn_shark()

    # 4. Rebuild sheets
    rebuild_sheets()


if __name__ == "__main__":
    main()
