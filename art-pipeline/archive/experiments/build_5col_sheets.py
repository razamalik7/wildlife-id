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
from test_fresh_20_batch2 import BATCH2_20_SPECIES, WORK, LORA, ARTIFACT_DIR, normalize_repaired_cutout
from test_pre_stylization_crop_repair import build_canny_from_repaired


def main() -> None:
    prev_dir = WORK / "clean_out_prev"
    prev_dir.mkdir(parents=True, exist_ok=True)

    changed_5 = {"01_collared_peccary", "02_ringtail", "17_spring_peeper", "19_horn_shark", "20_hallers_round_ray"}
    for stem, *_ in BATCH2_20_SPECIES:
        if stem not in changed_5:
            shutil.copyfile(WORK / "clean_out" / f"{stem}.png", prev_dir / f"{stem}.png")

    meta = json.loads((WORK / "batch2_metadata.json").read_text(encoding="utf-8"))

    cnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0-small", torch_dtype=torch.bfloat16, variant="fp16"
    )
    cnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small", torch_dtype=torch.bfloat16, variant="fp16"
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

    prev_specs = [
        (
            "01_collared_peccary",
            normalize_repaired_cutout(Image.open(ARTIFACT_DIR / "genai_b2_01_collared_peccary.png")),
            f"animl_gouache_style, Collared Peccary (Pecari tajacu), full body wildlife illustration, {meta['01_collared_peccary']['visual_traits']}, {STYLE_V6}",
            NEG_V6_UNIFIED,
            0.54,
            0.45,
        ),
        (
            "02_ringtail",
            Image.open(WORK / "cutouts_1024" / "02_ringtail.png").convert("RGB"),
            f"animl_gouache_style, Ringtail (Bassariscus astutus), full body wildlife illustration, {meta['02_ringtail']['visual_traits']}, {STYLE_V6}",
            NEG_V6_UNIFIED,
            0.54,
            0.45,
        ),
        (
            "17_spring_peeper",
            Image.open(WORK / "cutouts_1024" / "17_spring_peeper.png").convert("RGB"),
            f"animl_gouache_style, a small tan woodland treefrog, full body wildlife illustration, exact dorsal markings and golden-brown eye matching the reference photo, toe pads visible, {STYLE_V6}",
            NEG_V6_UNIFIED,
            0.47,
            0.50,
        ),
        (
            "19_horn_shark",
            Image.open(WORK / "cands_top3" / "19_horn_shark" / "wild_cut_1.png").convert("RGB"),
            f"animl_gouache_style, a brown spotted Heterodontus benthic shark (Heterodontus francisci), full body wildlife illustration, warm tan-brown body covered in distinct dark brown-black spots, blunt rounded snout with supraorbital ridges, two dorsal fins with pale anterior spines, broad pectoral fins, {STYLE_V6}",
            "horns, spikes on nose, whiskers, barbels, aquarium glass reflection, tiger stripes, " + NEG_V6_UNIFIED,
            0.52,
            0.48,
        ),
        (
            "20_hallers_round_ray",
            normalize_repaired_cutout(Image.open(ARTIFACT_DIR / "genai_ogref_20_hallers_round_ray_1793863422214.jpg")),
            f"animl_gouache_style, Haller's Round Ray (Urobatis halleri), full body wildlife illustration, nearly circular brown-and-ochre spotted pectoral disc, symmetrical pair of two small dark dorsal eyes and spiracles on top of head, stout tail with dark venomous spine and rounded caudal fin, {STYLE_V6}",
            "single eye, cyclops, missing second eye, " + NEG_V6_UNIFIED,
            0.50,
            0.50,
        ),
    ]

    raw_prev = {}
    for stem, cut_im, prompt, neg_p, strength, canny_w in prev_specs:
        canny_im = build_canny_from_repaired(cut_im)
        depth_im = Image.open(WORK / "depth_final" / f"{stem}.png").convert("RGB")
        pos_e, neg_e, pos_p, neg_p_emb = encode_prompt_sdxl_multichunk(pipe, prompt, neg_p, max_chunks=3)
        gen = torch.Generator(device="cuda").manual_seed(42)
        res_im = pipe(
            prompt_embeds=pos_e,
            negative_prompt_embeds=neg_e,
            pooled_prompt_embeds=pos_p,
            negative_pooled_prompt_embeds=neg_p_emb,
            image=cut_im,
            control_image=[canny_im, depth_im],
            strength=strength,
            controlnet_conditioning_scale=[canny_w, 0.25],
            control_guidance_start=0.0,
            control_guidance_end=[0.78, 0.60],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
        ).images[0]
        raw_prev[stem] = res_im

    del pipe, cnet_canny, cnet_depth
    gc.collect()
    torch.cuda.empty_cache()

    for stem, res_im in raw_prev.items():
        out_p = prev_dir / f"{stem}.png"
        if out_p.exists():
            out_p.unlink()
        clean_post(res_im, out_p)

    # Also apply the two-eyed circle dot from update_batch2_four onto prev_dir/20_hallers_round_ray.png so Col 4 matches the exact image shown in the previous message
    ray_prev_p = prev_dir / "20_hallers_round_ray.png"
    ray_rgba = np.array(Image.open(ray_prev_p).convert("RGBA"))
    rgb = ray_rgba[:, :, :3].copy()
    for ex, ey in [(249, 567), (281, 539)]:
        cv2.circle(rgb, (ex, ey), 6, (142, 112, 78), -1, lineType=cv2.LINE_AA)
        cv2.circle(rgb, (ex, ey), 4, (36, 30, 26), -1, lineType=cv2.LINE_AA)
        cv2.circle(rgb, (ex - 1, ey - 1), 1, (195, 185, 170), -1, lineType=cv2.LINE_AA)
    ray_rgba[:, :, :3] = rgb
    Image.fromarray(ray_rgba).save(ray_prev_p)

    for part_idx, slice_sp in [(1, BATCH2_20_SPECIES[:10]), (2, BATCH2_20_SPECIES[10:20])]:
        col_w, row_h, header_h = 310, 310, 56
        w = 5 * col_w
        h = header_h + len(slice_sp) * row_h
        sheet = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        try:
            font_hdr = ImageFont.truetype("arialbd.ttf", 19)
            font_lbl = ImageFont.truetype("arialbd.ttf", 16)
        except Exception:
            font_hdr = ImageFont.load_default()
            font_lbl = ImageFont.load_default()

        headers = [
            "1. Selected Ref Photo",
            "2. Pre-Stylization Cutout",
            "3. Canny Structure Map",
            "4. Previous Run (Before)",
            "5. Targeted Fix (After)",
        ]
        draw.rectangle([0, 0, w, header_h], fill=(35, 38, 45))
        for c_i, h_text in enumerate(headers):
            draw.text((c_i * col_w + 10, 16), h_text, fill=(255, 255, 255), font=font_hdr)

        for r_i, (stem, common, sci, _, _, _) in enumerate(slice_sp):
            y0 = header_h + r_i * row_h
            p_photo = WORK / "photos_selected" / f"{stem}.png"
            p_cut = WORK / "cutouts_final" / f"{stem}.png"
            p_canny = WORK / "canny_final" / f"{stem}.png"
            p_prev = prev_dir / f"{stem}.png"
            p_styl = WORK / "clean_out" / f"{stem}.png"
            for c_i, p in enumerate([p_photo, p_cut, p_canny, p_prev, p_styl]):
                x0 = c_i * col_w
                if p.exists():
                    im = Image.open(p)
                    if im.mode == "RGBA":
                        bg = Image.new("RGB", im.size, (255, 255, 255))
                        bg.paste(im, mask=im.split()[3])
                        im = bg
                    else:
                        im = im.convert("RGB")
                    sheet.paste(im.resize((col_w - 16, row_h - 40), Image.Resampling.LANCZOS), (x0 + 8, y0 + 32))
                draw.rectangle([x0, y0, x0 + col_w, y0 + row_h], outline=(220, 220, 220), width=1)
            draw.text((12, y0 + 8), f"{stem}: {common} ({sci})", fill=(20, 20, 20), font=font_lbl)

        out_sheet = ARTIFACT_DIR / f"v6_batch2_5col_part{part_idx}.jpg"
        sheet.save(out_sheet, quality=92)
        print(f"Saved 5-column sheet: {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
