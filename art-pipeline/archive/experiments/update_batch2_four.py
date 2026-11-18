"""
Re-render the 4 updated Batch 2 species based on user feedback:
1. `10_hooded_merganser`: Dual-reference `[Original Photo + Cutout]` GenAI repair (`genai_ogref_10_hooded_merganser_1793863390292.jpg`)
   where the submerged lower belly and rear paddling feet match the original photo's water plane and camera angle.
2. `17_spring_peeper`: De-bias from the memorized `spring_peeper_faithful.png` LoRA training anchor by removing the literal
   'Spring Peeper' trigger token, lowering denoising strength (0.48) and increasing Canny (0.50), and restoring the
   reference photo's exact golden-brown eye iris.
3. `19_horn_shark`: Replace the reflective aquarium-glass photo with the clean wild scuba photo (`wild_cut_1.png` / `wild_photo_1.png`)
   and also add `no_aquarium_glass_reflection` to the VLM vetter criteria.
4. `20_hallers_round_ray`: Dual-reference `[Original Photo + Cutout]` GenAI repair (`genai_ogref_20_hallers_round_ray_1793863422214.jpg`)
   with both symmetrical dorsal eyes and spiracles clearly restored and preserved through Canny + SDXL.
"""
import gc
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

sys.path.insert(0, str(Path(__file__).parent))
from test_fresh_20_e2e import STYLE_V6, NEG_V6_UNIFIED, clean_post, encode_prompt_sdxl_multichunk
from test_fresh_20_batch2 import WORK, LORA, ARTIFACT_DIR, normalize_repaired_cutout
from test_pre_stylization_crop_repair import build_canny_from_repaired
from fix_batch2_polish import rebuild_sheets, restore_eye_iris


def prepare_four_cutouts() -> None:
    # 1. 10_hooded_merganser (Dual-ref Original + Cutout GenAI repair)
    merg_src = ARTIFACT_DIR / "genai_ogref_10_hooded_merganser_1793863390292.jpg"
    merg_norm = normalize_repaired_cutout(Image.open(merg_src))
    merg_norm.save(WORK / "cutouts_final" / "10_hooded_merganser.png")

    # 2. 19_horn_shark (Wild non-aquarium scuba photo + cutout)
    wild_photo = WORK / "cands_top3" / "19_horn_shark" / "wild_photo_1.png"
    wild_cut = WORK / "cands_top3" / "19_horn_shark" / "wild_cut_1.png"
    shutil.copyfile(wild_photo, WORK / "photos_selected" / "19_horn_shark.png")
    shutil.copyfile(wild_cut, WORK / "cutouts_1024" / "19_horn_shark.png")
    shutil.copyfile(wild_cut, WORK / "cutouts_final" / "19_horn_shark.png")

    # 3. 20_hallers_round_ray (Dual-ref Original + Cutout GenAI repair with two dorsal eyes)
    ray_src = ARTIFACT_DIR / "genai_ogref_20_hallers_round_ray_1793863422214.jpg"
    ray_norm = normalize_repaired_cutout(Image.open(ray_src))
    ray_norm.save(WORK / "cutouts_final" / "20_hallers_round_ray.png")

    # Build Canny maps for all 4
    for stem in ["10_hooded_merganser", "17_spring_peeper", "19_horn_shark", "20_hallers_round_ray"]:
        cut_im = Image.open(WORK / "cutouts_final" / f"{stem}.png").convert("RGB")
        canny_im = build_canny_from_repaired(cut_im)
        canny_im.save(WORK / "canny_final" / f"{stem}.png")

    # Build Depth maps for all 4
    depth_estimator = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cuda",
    )
    for stem in ["10_hooded_merganser", "17_spring_peeper", "19_horn_shark", "20_hallers_round_ray"]:
        cut_im = Image.open(WORK / "cutouts_final" / f"{stem}.png").convert("RGB")
        fg = np.any(np.array(cut_im) < 244, axis=-1)
        d_im = depth_estimator(cut_im)["depth"].resize((1024, 1024), Image.Resampling.BILINEAR)
        d_np = np.array(d_im, dtype=np.float32)
        d_np[~fg] = 0.0
        if fg.any() and d_np[fg].max() > d_np[fg].min():
            d_np[fg] = (d_np[fg] - d_np[fg].min()) / (d_np[fg].max() - d_np[fg].min()) * 215.0 + 40.0
        d_u8 = d_np.astype(np.uint8)
        Image.fromarray(np.stack([d_u8, d_u8, d_u8], axis=-1)).save(WORK / "depth_final" / f"{stem}.png")

    del depth_estimator
    gc.collect()
    torch.cuda.empty_cache()


def render_four_sdxl() -> None:
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

    specs = [
        (
            "10_hooded_merganser",
            f"animl_gouache_style, Hooded Merganser (Lophodytes cucullatus), full body wildlife illustration, "
            f"fan-shaped white crest with black border, golden-yellow eye, warm rufous-cinnamon flanks, "
            f"tucked webbed feet under rear flank, {STYLE_V6}",
            NEG_V6_UNIFIED,
            0.54,
            0.46,
        ),
        (
            "17_spring_peeper",
            # Avoid the literal 'Spring Peeper' training-anchor caption token so the LoRA does not override
            # this reference photo's exact tan dorsal markings and golden-brown eye color!
            f"animl_gouache_style, a small tan woodland treefrog, full body wildlife illustration, "
            f"exact dorsal markings and golden-brown eye matching the reference photo, toe pads visible, {STYLE_V6}",
            NEG_V6_UNIFIED,
            0.47,
            0.50,
        ),
        (
            "19_horn_shark",
            f"animl_gouache_style, a brown spotted Heterodontus benthic shark (Heterodontus francisci), "
            f"full body wildlife illustration, warm tan-brown body covered in distinct dark brown-black spots, "
            f"blunt rounded snout with supraorbital ridges, two dorsal fins with pale anterior spines, "
            f"broad pectoral fins, {STYLE_V6}",
            "horns, spikes on nose, whiskers, barbels, aquarium glass reflection, tiger stripes, " + NEG_V6_UNIFIED,
            0.52,
            0.48,
        ),
        (
            "20_hallers_round_ray",
            f"animl_gouache_style, Haller's Round Ray (Urobatis halleri), full body wildlife illustration, "
            f"nearly circular brown-and-ochre spotted pectoral disc, symmetrical pair of two small dark dorsal eyes "
            f"and spiracles on top of head, stout tail with dark venomous spine and rounded caudal fin, {STYLE_V6}",
            "single eye, cyclops, missing second eye, " + NEG_V6_UNIFIED,
            0.50,
            0.50,
        ),
    ]

    for stem, prompt, neg_prompt, strength, canny_w in specs:
        cut_im = Image.open(WORK / "cutouts_final" / f"{stem}.png").convert("RGB")
        canny_im = Image.open(WORK / "canny_final" / f"{stem}.png").convert("RGB")
        depth_im = Image.open(WORK / "depth_final" / f"{stem}.png").convert("RGB")
        pos_e, neg_e, pos_p, neg_p = encode_prompt_sdxl_multichunk(pipe, prompt, neg_prompt, max_chunks=3)
        gen = torch.Generator(device="cuda").manual_seed(42)
        res_im = pipe(
            prompt_embeds=pos_e,
            negative_prompt_embeds=neg_e,
            pooled_prompt_embeds=pos_p,
            negative_pooled_prompt_embeds=neg_p,
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
        res_im.save(WORK / "raw_stylized" / f"{stem}.png")
        print(f"Rendered {stem}", flush=True)

    del pipe, cnet_canny, cnet_depth
    gc.collect()
    torch.cuda.empty_cache()


def post_process_four() -> None:
    for stem in ["10_hooded_merganser", "17_spring_peeper", "19_horn_shark", "20_hallers_round_ray"]:
        res_im = Image.open(WORK / "raw_stylized" / f"{stem}.png").convert("RGB")
        out_p = WORK / "clean_out" / f"{stem}.png"
        if out_p.exists():
            out_p.unlink()
        clean_post(res_im, out_p)
        shutil.copyfile(out_p, ARTIFACT_DIR / f"batch2_{stem}.png")

    # Auto-detect the golden eye box inside the head region of `10_hooded_merganser`
    merg_cut = np.array(Image.open(WORK / "cutouts_final" / "10_hooded_merganser.png").convert("RGB"))
    # Head region is y=180..480, x=600..850
    head_roi = merg_cut[180:480, 600:850]
    golden_mask = (
        (head_roi[:, :, 0].astype(int) > 175)
        & (head_roi[:, :, 1].astype(int) > 130)
        & (head_roi[:, :, 2].astype(int) < 95)
        & ((head_roi[:, :, 0].astype(int) - head_roi[:, :, 2].astype(int)) > 90)
    )
    ys, xs = np.where(golden_mask)
    if len(ys) > 5:
        yc = int(np.median(ys)) + 180
        xc = int(np.median(xs)) + 600
        eye_box = (xc - 24, yc - 22, xc + 24, yc + 22)
        print(f"10_hooded_merganser detected eye_box: {eye_box}", flush=True)
        restore_eye_iris("10_hooded_merganser", eye_box, min_cut_l=105.0)

    # Restore exact reference-photo eye iris on `17_spring_peeper`
    restore_eye_iris("17_spring_peeper", (778, 394, 852, 456), min_cut_l=85.0)

    # Ensure both dorsal eyes on `20_hallers_round_ray` remain crisp and clearly paired
    ray_cut = np.array(Image.open(WORK / "cutouts_final" / "20_hallers_round_ray.png").convert("RGB"))
    ray_out_p = WORK / "clean_out" / "20_hallers_round_ray.png"
    ray_styl = np.array(Image.open(ray_out_p).convert("RGBA"))
    # Blend a cel-smoothed version of the dorsal head/eye region (x=180..360, y=410..560) where the dark eyes sit
    ray_smooth = cv2.bilateralFilter(ray_cut, d=9, sigmaColor=40, sigmaSpace=40)
    eye_roi_cut = ray_smooth[410:560, 180:360]
    eye_roi_styl = ray_styl[410:560, 180:360, :3]
    cut_l = cv2.cvtColor(eye_roi_cut, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    # Dark eye pupil/iris spots in the repaired cutout have L < 85
    dark_eye_w = np.clip((92.0 - cut_l) / 30.0, 0.0, 1.0)
    dark_eye_w = cv2.GaussianBlur(dark_eye_w, (5, 5), 1.0)[:, :, None]
    ray_styl[410:560, 180:360, :3] = np.clip(
        eye_roi_styl.astype(np.float32) * (1.0 - 0.85 * dark_eye_w)
        + eye_roi_cut.astype(np.float32) * (0.85 * dark_eye_w),
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(ray_styl).save(ray_out_p)
    shutil.copyfile(ray_out_p, ARTIFACT_DIR / "batch2_20_hallers_round_ray.png")


if __name__ == "__main__":
    prepare_four_cutouts()
    render_four_sdxl()
    post_process_four()
    rebuild_sheets()
