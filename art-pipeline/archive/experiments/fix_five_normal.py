"""
Targeted normal fix pass for the 5 Batch 2 species:
1. `01_collared_peccary`: Restore the signature pale yellowish-white diagonal shoulder collar stripe in the cutout + Canny + SDXL.
2. `02_ringtail`: Eliminate dorsal back stripes by removing 'banded/ringed' token bleeding from the body and suppressing internal back Canny noise.
3. `17_spring_peeper`: Lower `lora_scale=0.60` and `strength=0.38` (`Canny=0.56`) so the overfit `spring_peeper.png` training image in `img_4x` does not override the reference photo's eye color or dorsal pattern.
4. `19_horn_shark`: Use the dual-reference GenAI-repaired cutout (`genai_fixed_19_horn_shark_1793864872455.jpg` with clean first dorsal fin and no reef rock lump) and omit the literal word 'shark' from the positive prompt (`strength=0.44, Canny=0.54`) so SDXL never draws a Great White shark face/teeth.
5. `20_hallers_round_ray`: Use the dual-reference GenAI-repaired cutout (`genai_fixed_20_hallers_round_ray_1793864962579.jpg` with natural 3/4-perspective raised batoid eyes and no central spot) with `strength=0.44, Canny=0.54`.
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


def prepare_five_cutouts() -> None:
    # 1. 01_collared_peccary: accentuate the pale yellowish-white shoulder collar band on the repaired cutout
    pec_src = ARTIFACT_DIR / "genai_b2_01_collared_peccary.png"
    pec_im = normalize_repaired_cutout(Image.open(pec_src))
    pec_arr = np.array(pec_im).astype(np.float32)
    fg_pec = np.any(pec_arr < 242, axis=-1)
    # Draw a soft natural diagonal collar mask from lower chest/throat (665, 535) up across the shoulder to (565, 345)
    collar_mask = np.zeros(pec_arr.shape[:2], dtype=np.float32)
    pts = np.array([[668, 540], [648, 495], [610, 425], [572, 355]], dtype=np.int32)
    cv2.polylines(collar_mask, [pts], isClosed=False, color=1.0, thickness=28, lineType=cv2.LINE_AA)
    collar_mask = cv2.GaussianBlur(collar_mask, (31, 31), 9.0) * fg_pec.astype(np.float32)
    # Brighten and warm the grizzled hairs along the collar stripe into a distinct cream-buff collar band
    target_collar = np.clip(pec_arr * 1.45 + np.array([58.0, 52.0, 36.0]), 0, 232)
    pec_arr = np.clip(
        pec_arr * (1.0 - 0.68 * collar_mask[:, :, None]) + target_collar * (0.68 * collar_mask[:, :, None]),
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(pec_arr).save(WORK / "cutouts_final" / "01_collared_peccary.png")

    # 2. 19_horn_shark: use genai_fixed_19_horn_shark_1793864872455.jpg
    shark_src = ARTIFACT_DIR / "genai_fixed_19_horn_shark_1793864872455.jpg"
    shark_norm = normalize_repaired_cutout(Image.open(shark_src))
    shark_norm.save(WORK / "cutouts_final" / "19_horn_shark.png")

    # 3. 20_hallers_round_ray: use genai_fixed_20_hallers_round_ray_1793864962579.jpg
    ray_src = ARTIFACT_DIR / "genai_fixed_20_hallers_round_ray_1793864962579.jpg"
    ray_norm = normalize_repaired_cutout(Image.open(ray_src))
    ray_norm.save(WORK / "cutouts_final" / "20_hallers_round_ray.png")

    # Build Canny maps for all 5
    for stem in ["01_collared_peccary", "02_ringtail", "17_spring_peeper", "19_horn_shark", "20_hallers_round_ray"]:
        cut_im = Image.open(WORK / "cutouts_final" / f"{stem}.png").convert("RGB")
        canny_im = build_canny_from_repaired(cut_im)
        if stem == "02_ringtail":
            # Suppress internal Canny edges inside the torso/back of the Ringtail (x=370..680, y=380..500)
            # while keeping the outer silhouette contour and the ringed tail + face 100% intact!
            c_arr = np.array(canny_im)
            fg = np.any(np.array(cut_im) < 242, axis=-1).astype(np.uint8) * 255
            inner_fg = cv2.erode(fg, np.ones((15, 15), np.uint8)) > 0
            torso_box = np.zeros_like(inner_fg)
            torso_box[380:515, 365:690] = True
            suppress = inner_fg & torso_box
            c_arr[suppress] = 0
            canny_im = Image.fromarray(c_arr)
        canny_im.save(WORK / "canny_final" / f"{stem}.png")

    # Build Depth maps for the updated cutouts
    depth_estimator = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cuda",
    )
    for stem in ["01_collared_peccary", "19_horn_shark", "20_hallers_round_ray"]:
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


def render_five_sdxl() -> None:
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
    pipe.to("cuda")

    specs = [
        (
            "01_collared_peccary",
            1.0,
            f"animl_gouache_style, Collared Peccary (Pecari tajacu), full body wildlife illustration, "
            f"coarse grizzled dark grey-brown bristle coat with a distinct pale yellowish-white collar stripe "
            f"across the shoulder, pig-like snout, slender legs with hooves, {STYLE_V6}",
            NEG_V6_UNIFIED,
            0.52,
            0.48,
        ),
        (
            "02_ringtail",
            0.85,
            # Avoid the words 'banded', 'ringed', or 'stripes' so the LoRA never puts armadillo/raccoon bands on the back!
            f"animl_gouache_style, Bassariscus astutus, slender desert mammal, full body wildlife illustration, "
            f"smooth uniform unpatterned buffy-grey-brown fur across the entire back and shoulders, "
            f"pointed fox-like face with white eye rings, long bushy black and white tail, {STYLE_V6}",
            "stripes on back, banded back, striped body, tabby stripes, armadillo bands, raccoon stripes, " + NEG_V6_UNIFIED,
            0.46,
            0.50,
        ),
        (
            "17_spring_peeper",
            0.55,
            # Use lower lora_scale (0.55) + lower denoising strength (0.38) so the single memorized frog in v6
            # does not override the reference photo's exact tan pattern or golden-brown eye!
            f"animl_gouache_style, small light tan amphibian, full body wildlife illustration, "
            f"warm golden-tan skin with distinct brown markings matching the photo, bright golden-brown iris, {STYLE_V6}",
            NEG_V6_UNIFIED,
            0.38,
            0.56,
        ),
        (
            "19_horn_shark",
            0.85,
            # Omit the literal word 'shark' so SDXL base does not hallucinate a Great White shark face/teeth!
            f"animl_gouache_style, Heterodontus francisci, benthic marine vertebrate, full body wildlife illustration, "
            f"blunt rounded pig-like snout with smooth closed ventral mouth, prominent supraorbital ridges above eye, "
            f"tall triangular first dorsal fin with pale anterior spine, warm brown skin covered in dark brown spots, "
            f"broad paddle-like pectoral fins, {STYLE_V6}",
            "shark teeth, open mouth, great white shark, grin, pointed snout, extra dorsal hump, " + NEG_V6_UNIFIED,
            0.44,
            0.54,
        ),
        (
            "20_hallers_round_ray",
            0.85,
            f"animl_gouache_style, Urobatis halleri, round benthic marine ray, full body wildlife illustration, "
            f"circular grey-brown and yellow-ochre spotted pectoral disc, two symmetrical raised dorsal eyes and spiracles "
            f"on anterior head, stout tail with rounded caudal fin, {STYLE_V6}",
            "single eye, cyclops, central white ring on head, " + NEG_V6_UNIFIED,
            0.44,
            0.54,
        ),
    ]

    for stem, lora_s, prompt, neg_prompt, strength, canny_w in specs:
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
            control_guidance_end=[0.80, 0.60],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
            cross_attention_kwargs={"scale": lora_s},
        ).images[0]
        res_im.save(WORK / "raw_stylized" / f"{stem}.png")
        print(f"Rendered {stem} (lora_scale={lora_s}, strength={strength})", flush=True)

    del pipe, cnet_canny, cnet_depth
    gc.collect()
    torch.cuda.empty_cache()


def post_process_five() -> None:
    for stem in ["01_collared_peccary", "02_ringtail", "17_spring_peeper", "19_horn_shark", "20_hallers_round_ray"]:
        res_im = Image.open(WORK / "raw_stylized" / f"{stem}.png").convert("RGB")
        out_p = WORK / "clean_out" / f"{stem}.png"
        if out_p.exists():
            out_p.unlink()
        clean_post(res_im, out_p)
        shutil.copyfile(out_p, ARTIFACT_DIR / f"batch2_{stem}.png")

    # Restore exact reference-photo golden-brown iris on `17_spring_peeper`
    restore_eye_iris("17_spring_peeper", (778, 394, 852, 456), min_cut_l=80.0)


if __name__ == "__main__":
    prepare_five_cutouts()
    render_five_sdxl()
    post_process_five()
    rebuild_sheets()
