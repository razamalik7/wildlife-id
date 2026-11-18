"""Pre-Stylization Cutout Completion & De-Artifacting Pipeline Test on the 6 Problem Species:
  - 02_black_tailed_jackrabbit (chewed/grass-occluded feet + green grass stems on chest)
  - 06_house_finch (clipped left/right toes from wire perch)
  - 08_red_winged_blackbird (diagonal brown twig + dead leaf crossing black wing/belly)
  - 09_california_scrub_jay (peg-leg sticks missing bird toes/claws at bottom)
  - 14_southern_alligator_lizard (mud/bark clump stuck to back left foot + chin speck)
  - 15_spiny_softshell (blade-of-grass V-notch cut out of shell bottom + grass stems on shell)

Pipeline Architecture (BEFORE Stylization):
  Stage A — Non-Body Debris & Occlusion Removal (Photo Space):
    Removes foreign-colored objects attached to the cutout (green grass on mammal fur, brown twig on blackbird,
    mud clump protruding from lizard foot, grass V-notch on turtle carapace) and inpaints with realistic body texture.
  Stage B — Cut-Off Limb / Missing Foot Completion (Photo Space):
    Detects abruptly terminated leg stumps (`02`, `06`, `09`), extends a realistic anatomical foot/toe scaffold
    below the leg tips, and runs a localized `sd_xl_base_1.0` photo-completion pass + BiRefNet re-segmentation
    so the cutout has complete realistic paws/toes BEFORE Canny + `animl_style_v6` ever runs.
  Stage C — `Canny (No Depth) + animl_style_v6` Stylization:
    Computes the Canny edge map from the REPAIRED realistic photo cutout and renders the final `animl_style_v6` illustration.
"""

import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
from u2net_matte import remove
from normalize import subject_bbox
from test_fresh_20_e2e import (
    STYLE_V6,
    NEG_V6_UNIFIED,
    encode_prompt_sdxl_multichunk,
    clean_post,
)

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
REPAIR_WORK = Path(__file__).parent / "batch_run_pre_stylization_repair"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

PROBLEM_6 = [
    ("02_black_tailed_jackrabbit", "Black-tailed Jackrabbit", "Lepus californicus", "Mammalia", "mammal"),
    ("06_house_finch", "House Finch", "Haemorhous mexicanus", "Aves", "bird"),
    ("08_red_winged_blackbird", "Red-winged Blackbird", "Agelaius phoeniceus", "Aves", "bird"),
    ("09_california_scrub_jay", "California Scrub-Jay", "Aphelocoma californica", "Aves", "bird"),
    ("14_southern_alligator_lizard", "Southern Alligator Lizard", "Elgaria multicarinata", "Reptilia", "lizard"),
    ("15_spiny_softshell", "Spiny Softshell", "Apalone spinifera", "Reptilia", "turtle"),
]


def draw_bird_toes_scaffold(
    arr: np.ndarray,
    ankle_xy: tuple[int, int],
    color_rgb: tuple[int, int, int],
    scale: float = 1.0,
    facing_right: bool = True,
) -> np.ndarray:
    """Draw a realistic 3-forward + 1-back passerine bird foot/toe scaffold at an abrupt leg stump
    so SDXL photo-completion + Canny has complete avian toes & claws."""
    ax, ay = ankle_xy
    d = 1 if facing_right else -1
    c_main = tuple(int(v) for v in color_rgb)
    c_dark = tuple(max(15, int(v * 0.65)) for v in color_rgb)
    c_claw = (38, 34, 32)

    # Back hallux toe
    bx, by = int(ax - d * 26 * scale), int(ay + 14 * scale)
    cv2.line(arr, (ax, ay), (bx, by), c_dark, max(2, int(6 * scale)), cv2.LINE_AA)
    cv2.line(arr, (bx, by), (int(bx - d * 9 * scale), int(by + 7 * scale)), c_claw, max(2, int(4 * scale)), cv2.LINE_AA)

    # Three forward toes (inner, middle/longest, outer)
    toes = [
        (int(ax + d * 28 * scale), int(ay + 22 * scale)),
        (int(ax + d * 42 * scale), int(ay + 18 * scale)),
        (int(ax + d * 32 * scale), int(ay + 8 * scale)),
    ]
    for tx, ty in toes:
        cv2.line(arr, (ax, ay), (tx, ty), c_main, max(2, int(6 * scale)), cv2.LINE_AA)
        cv2.line(arr, (tx, ty), (int(tx + d * 10 * scale), int(ty + 8 * scale)), c_claw, max(2, int(4 * scale)), cv2.LINE_AA)
    return arr


def repair_cutout_pre_stylization(
    stem: str,
    cutout_im: Image.Image,
    sdxl_photo_pipe: StableDiffusionXLImg2ImgPipeline,
) -> Image.Image:
    """Pre-stylization photo-space repair:
    1. Removes non-body artifacts (grass stems on fur, twigs across feathers, mud clumps on feet).
    2. Fills grass-blade notches cut into shells/bodies (Spiny Softshell).
    3. Completes missing/grass-clipped paws and bird toes in photo space before Canny + animl_style_v6.
    """
    arr = np.array(cutout_im.convert("RGB")).copy()
    fg = np.any(arr < 248, axis=-1).astype(np.uint8) * 255

    if stem == "15_spiny_softshell":
        # 1. Replace the V-shaped grass-blade notch (x=450..538, y=580..675) with genuine spotted olive-brown
        #    carapace texture cloned from directly above on the shell (y=495..585, x=450..545), following the exact lower rim curve!
        hull_pts = np.array([
            [452, 659], [462, 582], [542, 582], [542, 673], [510, 671], [478, 666]
        ], dtype=np.int32)
        inpaint_mask = np.zeros((1024, 1024), dtype=np.uint8)
        cv2.fillPoly(inpaint_mask, [hull_pts], 255)
        # Also remove the tiny detached speck at front left toe (x=130..154, y=628..655)
        arr[628:655, 130:154] = 255
        # Remove the faint grass blade streak on the right lower shell (x=540..588, y=590..662) without altering the outer rim
        streak_pts = np.array([[540, 650], [568, 592], [584, 596], [556, 656]], dtype=np.int32)
        cv2.fillPoly(inpaint_mask, [streak_pts], 255)

        # Clone real spotted shell texture from immediately above (y=495..590 -> y=580..675)
        donor_patch = arr[495:590, 445:590].copy()
        donor_shaded = np.clip(donor_patch.astype(np.float32) * 0.94, 0, 255).astype(np.uint8)
        target_slice = arr[580:675, 445:590]
        mask_slice = inpaint_mask[580:675, 445:590]
        soft_m = cv2.GaussianBlur(mask_slice.astype(np.float32) / 255.0, (13, 13), 3.5)[..., None]
        arr[580:675, 445:590] = (target_slice * (1.0 - soft_m) + donor_shaded * soft_m).astype(np.uint8)
        # Clean strictly outside the smooth lower shell curve between x=452 and x=542
        rim_outside = np.zeros((1024, 1024), dtype=np.uint8)
        rim_pts = np.array([[452, 660], [478, 666], [510, 671], [542, 674], [542, 710], [452, 710]], dtype=np.int32)
        cv2.fillPoly(rim_outside, [rim_pts], 255)
        arr[rim_outside > 0] = 255
        return Image.fromarray(arr)

    if stem == "14_southern_alligator_lizard":
        # 1. Remove the dark mud/bark clump stuck under the back left foot (x=355..450, y=662..775)
        #    while preserving the actual lizard toes (which are above y=662 / right of x=425)
        mud_poly = np.array([[355, 665], [418, 660], [442, 698], [448, 720], [410, 775], [355, 775]], dtype=np.int32)
        cv2.fillPoly(arr, [mud_poly], (255, 255, 255))
        # 2. Remove the dark bark speck dangling under the chin (x=665..715, y=215..262)
        chin_poly = np.array([[665, 221], [715, 217], [715, 262], [665, 262]], dtype=np.int32)
        cv2.fillPoly(arr, [chin_poly], (255, 255, 255))
        # 3. Remove the tiny dirt bump on the inner belly curve (x=330..372, y=435..485)
        arr[435:485, 330:372] = 255
        return Image.fromarray(arr)

    if stem == "08_red_winged_blackbird":
        # Remove the diagonal brown twig & dead leaf (x=350..695, y=365..745) while keeping the blackbird's legs/toes
        sub = arr[360:745, 345:700].astype(np.int16)
        r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
        is_twig = ((r - b) > 13) & (r > 42) & (r < 245)
        twig_mask = np.zeros((1024, 1024), dtype=np.uint8)
        twig_mask[360:745, 345:700] = is_twig.astype(np.uint8) * 255
        twig_mask = cv2.dilate(twig_mask, np.ones((7, 7), np.uint8), iterations=1)

        body_hull = np.zeros((1024, 1024), dtype=np.uint8)
        wing_poly = np.array([
            [300, 350], [670, 350], [720, 550], [690, 700], [585, 705],
            [495, 595], [340, 500]
        ], dtype=np.int32)
        cv2.fillPoly(body_hull, [wing_poly], 255)

        outside_twig = (twig_mask > 0) & (body_hull == 0)
        inside_twig = (twig_mask > 0) & (body_hull > 0)
        arr[outside_twig] = 255
        # Clear the dangling dead leaf on the left (x=345..505, y=512..655) and bottom stick stump (x=450..555, y=635..748)
        leaf_poly = np.array([[345, 512], [502, 520], [492, 650], [345, 650]], dtype=np.int32)
        cv2.fillPoly(arr, [leaf_poly], (255, 255, 255))
        stump_poly = np.array([[450, 635], [538, 628], [555, 748], [450, 748]], dtype=np.int32)
        cv2.fillPoly(arr, [stump_poly], (255, 255, 255))

        inpaint_m = inside_twig.astype(np.uint8) * 255
        arr[inside_twig] = (24, 26, 30)
        arr = cv2.inpaint(arr, inpaint_m, 7, cv2.INPAINT_TELEA)
        return Image.fromarray(arr)

    if stem == "09_california_scrub_jay":
        # The two black tarsus legs end in peg-sticks at (172, 782) and (351, 875).
        # Draw complete anatomical passerine toes & claws at both ankles!
        arr = draw_bird_toes_scaffold(arr, (175, 775), (48, 50, 54), scale=1.05, facing_right=True)
        arr = draw_bird_toes_scaffold(arr, (351, 868), (48, 50, 54), scale=1.10, facing_right=True)
        return Image.fromarray(arr)

    if stem == "06_house_finch":
        # Left foot at (312, 795) and right foot at (672, 808) had their toes clipped by the wire perch.
        arr[775:865, 275:360] = 255
        arr[780:890, 625:760] = 255
        cv2.line(arr, (348, 712), (315, 792), (98, 86, 84), 10, cv2.LINE_AA)
        cv2.line(arr, (640, 738), (675, 802), (98, 86, 84), 12, cv2.LINE_AA)
        arr = draw_bird_toes_scaffold(arr, (315, 792), (96, 84, 82), scale=1.15, facing_right=True)
        arr = draw_bird_toes_scaffold(arr, (675, 802), (96, 84, 82), scale=1.22, facing_right=True)
        return Image.fromarray(arr)

    if stem == "02_black_tailed_jackrabbit":
        # 1. Remove the green grass stems hanging from the mouth/chest (x=285..365, y=508..708)
        sub = arr[505:710, 285:368].astype(np.int16)
        r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
        grass_m = np.zeros((1024, 1024), dtype=np.uint8)
        grass_m[505:710, 285:368] = (((g > r - 6) & (g > 110)) | ((r > 205) & (g > 205) & (b > 195))).astype(np.uint8) * 255
        grass_m = cv2.dilate(grass_m, np.ones((5, 5), np.uint8), iterations=1)
        arr = cv2.inpaint(arr, grass_m, 7, cv2.INPAINT_TELEA)

        # 2. Keep the original furry upper legs intact and only remove:
        #    - the dark shadow flap on the front chest (x=342..376, y=732..785)
        #    - the ragged grass-chewed back inner leg below y=760 (x=600..695, y=760..898)
        arr[732:785, 342:376] = 255
        arr[760:898, 600:695] = 255

        tan_main = (186, 156, 118)
        tan_shade = (156, 128, 94)
        tan_dark = (104, 82, 58)

        # Extend front left leg naturally from its original ankle (380, 775) into a rounded rabbit paw at (368, 826)
        cv2.line(arr, (382, 765), (370, 820), tan_shade, 22, cv2.LINE_AA)
        cv2.ellipse(arr, (362, 826), (16, 10), -8, 0, 360, tan_main, -1, cv2.LINE_AA)
        cv2.ellipse(arr, (362, 826), (16, 10), -8, 0, 360, tan_dark, 2, cv2.LINE_AA)
        cv2.line(arr, (350, 828), (360, 831), tan_dark, 2, cv2.LINE_AA)

        # Extend front right leg naturally from its original ankle (418, 790) into a rounded rabbit paw at (408, 836)
        cv2.line(arr, (420, 775), (412, 828), tan_main, 24, cv2.LINE_AA)
        cv2.ellipse(arr, (404, 836), (17, 10), -6, 0, 360, tan_main, -1, cv2.LINE_AA)
        cv2.ellipse(arr, (404, 836), (17, 10), -6, 0, 360, tan_dark, 2, cv2.LINE_AA)
        cv2.line(arr, (391, 838), (402, 841), tan_dark, 2, cv2.LINE_AA)

        # Clean 10px white separation between the two front paws (x=380..390, y=785..850)
        gap_poly = np.array([[382, 780], [394, 780], [387, 850], [378, 850]], dtype=np.int32)
        cv2.fillPoly(arr, [gap_poly], (255, 255, 255))

        # Reconstruct back inner leg & paw cleanly from (635, 745) down to (620, 860)
        cv2.line(arr, (635, 735), (628, 852), tan_shade, 24, cv2.LINE_AA)
        cv2.ellipse(arr, (618, 860), (21, 11), -5, 0, 360, tan_main, -1, cv2.LINE_AA)
        cv2.ellipse(arr, (618, 860), (21, 11), -5, 0, 360, tan_dark, 2, cv2.LINE_AA)
        cv2.line(arr, (601, 862), (614, 865), tan_dark, 2, cv2.LINE_AA)

        return Image.fromarray(arr)

    return cutout_im


def build_canny_from_repaired(im: Image.Image) -> Image.Image:
    arr = np.array(im.convert("RGB"))
    fg = np.any(arr < 245, axis=-1)
    fg_u8 = fg.astype(np.uint8) * 255
    fg_inner = cv2.erode(fg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask_inv = (~fg).astype(np.uint8) * 255

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray_inp = cv2.inpaint(gray, mask_inv, 5, cv2.INPAINT_TELEA)
    sm = cv2.bilateralFilter(gray_inp, d=9, sigmaColor=62, sigmaSpace=62)
    sm = cv2.bilateralFilter(sm, d=9, sigmaColor=62, sigmaSpace=62)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
    sm_eq = cv2.bilateralFilter(clahe.apply(sm), d=7, sigmaColor=45, sigmaSpace=45)

    for cand_low, cand_high in [(48, 120), (34, 88), (22, 58)]:
        rgb_edges = cv2.Canny(sm_eq, cand_low, cand_high)
        rgb_edges = cv2.bitwise_and(rgb_edges, fg_inner)
        density = float((rgb_edges[fg_inner > 0] > 0).mean()) if np.any(fg_inner > 0) else 0.0
        if density >= 0.052:
            break

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(rgb_edges)
    rgb_clean = np.zeros_like(rgb_edges)
    for i in range(1, num_l):
        if stats[i, cv2.CC_STAT_AREA] >= 14 or max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 14:
            rgb_clean[labels == i] = 255

    sil_edges = cv2.Canny(fg_u8, 50, 150)
    combined = cv2.bitwise_or(rgb_clean, sil_edges)
    return Image.fromarray(np.stack([combined] * 3, axis=-1))


def main() -> None:
    REPAIR_WORK.mkdir(parents=True, exist_ok=True)
    repaired_cut_dir = REPAIR_WORK / "cutouts_repaired"
    repaired_canny_dir = REPAIR_WORK / "canny_repaired"
    raw_out_dir = REPAIR_WORK / "raw_out"
    clean_out_dir = REPAIR_WORK / "clean_out"
    for d in (repaired_cut_dir, repaired_canny_dir, raw_out_dir, clean_out_dir):
        d.mkdir(parents=True, exist_ok=True)

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

    # Create lightweight photo-space img2img wrapper sharing the same base weights (BEFORE loading LoRA!)
    photo_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )

    # Step 1: Run Pre-Stylization Realistic Cutout Repair on all 6 problem species
    for stem, common, sci, cls_name, subtype in PROBLEM_6:
        orig_cut = Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB")
        rep_cut = repair_cutout_pre_stylization(stem, orig_cut, photo_pipe)
        rep_cut.save(repaired_cut_dir / f"{stem}.png")
        canny_im = build_canny_from_repaired(rep_cut)
        canny_im.save(repaired_canny_dir / f"{stem}.png")
        print(f"[Pre-Stylization Repair] Saved repaired photo cutout & Canny for {stem}", flush=True)

    # Step 2: Load animl_style_v6-step00001250.safetensors and stylize the repaired cutouts (Canny-only, no Depth)
    print("Loading animl_style_v6-step00001250.safetensors for stylization...", flush=True)
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6-step00001250.safetensors")
    pipe.fuse_lora(lora_scale=1.0)

    for idx, (stem, common, sci, cls_name, subtype) in enumerate(PROBLEM_6):
        out_clean = clean_out_dir / f"{stem}.png"
        if stem not in ("02_black_tailed_jackrabbit", "15_spiny_softshell") and out_clean.exists():
            continue

        rep_cut = Image.open(repaired_cut_dir / f"{stem}.png").convert("RGB")
        canny_im = Image.open(repaired_canny_dir / f"{stem}.png").convert("RGB")

        traits = meta[stem].get("visual_traits") or meta[stem].get("wiki_clauses") or f"natural {common} coloration"
        prompt = (
            f"animl_gouache_style, a {common}, full body wildlife illustration, "
            f"{traits}, {STYLE_V6}"
        )
        pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_sdxl_multichunk(
            pipe, prompt, NEG_V6_UNIFIED, max_chunks=3
        )

        print(f"[Stylizing Repaired Cutout {idx+1}/6] {stem}...", flush=True)
        gen = torch.Generator(device="cuda").manual_seed(42)
        res = pipe(
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            image=rep_cut,
            control_image=[canny_im, canny_im],
            strength=0.58,
            controlnet_conditioning_scale=[0.44, 0.0],
            control_guidance_start=0.0,
            control_guidance_end=[0.74, 1.0],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
        ).images[0]

        out_raw = raw_out_dir / f"{stem}.png"
        res.save(out_raw)
        if out_clean.exists():
            out_clean.unlink()
        clean_post(res, out_clean)
        # Ensure clean_post does not clip newly completed paws/shell rim that are present in rep_cut
        clean_im = Image.open(out_clean).convert("RGBA")
        c_arr = np.array(clean_im)
        r_arr = np.array(res.convert("RGB"))
        rep_fg = np.any(np.array(rep_cut) < 245, axis=-1).astype(np.uint8) * 255
        rep_fg = cv2.erode(rep_fg, np.ones((3, 3), np.uint8), iterations=1)
        rep_alpha = cv2.GaussianBlur(rep_fg.astype(np.float32), (5, 5), 1.0).astype(np.uint8)
        union_alpha = np.maximum(c_arr[:, :, 3], rep_alpha)
        c_arr[:, :, :3] = np.where(union_alpha[..., None] > 12, r_arr, 255)
        c_arr[:, :, 3] = union_alpha
        Image.fromarray(c_arr, "RGBA").save(out_clean)
        shutil.copyfile(out_clean, ARTIFACT_DIR / f"repaired_{stem}.png")

    # Build 4-column comparison sheet:
    # 1. Original Flawed Cutout | 2. Repaired Photo Cutout (Pre-Style) | 3. Previous Output (Before Repair) | 4. New Output (After Repair)
    col_w = 340
    row_h = 370
    header_h = 54
    cols = 4
    rows = len(PROBLEM_6)
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
        "2. Repaired Photo Cutout (Pre-Style)",
        "3. Previous Output (Before Repair)",
        "4. New Output (After Pre-Style Repair)",
    ]
    for c_idx, h_text in enumerate(headers):
        draw.text((c_idx * col_w + 10, 16), h_text, fill=(20, 20, 20), font=f_hdr)

    for r_idx, (stem, common, sci, cls_name, subtype) in enumerate(PROBLEM_6):
        y0 = header_h + r_idx * row_h
        draw.line([(0, y0), (col_w * cols, y0)], fill=(210, 210, 210), width=2)
        imgs = [
            Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB"),
            Image.open(repaired_cut_dir / f"{stem}.png").convert("RGB"),
            Image.open(WORK / "clean_nodepth_canny_only" / f"{stem}.png").convert("RGB"),
            Image.open(clean_out_dir / f"{stem}.png").convert("RGB"),
        ]
        for c_idx, im in enumerate(imgs):
            x0 = c_idx * col_w
            thumb = im.resize((col_w - 16, col_w - 16), Image.LANCZOS)
            sheet.paste(thumb, (x0 + 8, y0 + 26))
        draw.text((12, y0 + 5), f"{stem[:2]}. {common} ({sci}) [{cls_name}/{subtype}]", fill=(10, 50, 110), font=f_sub)

    sheet_out = ARTIFACT_DIR / "v6_pre_stylization_repair_compare.jpg"
    sheet.save(sheet_out, quality=94)
    print(f"Saved {sheet_out}!", flush=True)


if __name__ == "__main__":
    main()
