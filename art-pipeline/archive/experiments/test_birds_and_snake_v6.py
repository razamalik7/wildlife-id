"""Targeted v6 Test (v2) for the 4 Waterfowl & Coiled Common Watersnake:
1. Waterfowl (Shoveler, Pintail, Wigeon, Teal):
   - Option B seamless rounded lower belly + webbed feet on Pintail and Wigeon (no waterline cutoff!).
   - Bold 3x stronger overlapping feather-contour scallops (-42 lum contour / +18 lum feather body) +
     removes 'flat cel-shaded' from STYLE_BIRD_V6 so SDXL renders layered American-Robin-style feather texture
     instead of smooth featureless blocks.
   - Keeps Green-winged Teal's gray flank unblurred (so its flank stays gray, not white) while prompting layered feathers instead of stippled dots.
2. Coiled Common Watersnake (278839486 and 15458841):
   - Single-polygon seamless tapered pointed tail tip (tapering down to a sharp point at (485, 592) and (415, 572),
     well separated from the head so SDXL never fuses the loop or draws a second head on a blunt tail).
"""

import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
PY = LORA / "venv" / "Scripts" / "python.exe"
V5_EVAL = Path(__file__).parent / "batch_run_v5_v2"
RAW_SHARP = Path(__file__).parent / "batch_run_raw_unblurred_6sp" / "inputs_sharp"
PRE_V6_DIR = Path(__file__).parent / "batch_run_pre_v6_diag"
EVAL_DIR = Path(__file__).parent / "batch_run_v6_eval"
TEST_DIR = Path(__file__).parent / "batch_run_birds_snake_v6_v2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

# Bird-specific style clause: NO 'flat cel-shaded' (which smoothed out interior feathers!);
# explicitly drives American-Robin / Wood-Duck layered gouache feather strokes & ink feather contour lines.
STYLE_BIRD_V6 = (
    "muted matte natural colors, hand-painted gouache plumage with distinct overlapping feather contour lines, "
    "textured wing and flank feathers with warm brown-umber ink feather strokes, stepped matte shading, "
    "zero gloss or shine, clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

STYLE_SNAKE_V6 = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "crisp scale pattern and dark charcoal-umber contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

NEG_V6 = (
    "photo, photorealistic, 3d render, 3d lighting, specular shine, glossy, shiny, specular highlight, plastic, airbrushed, "
    "smooth plastic body, featureless plumage, stippled dots, pointillism, "
    "watercolor, painterly brush swirls, oil painting, heavy black sticker outline, "
    "branch, twig, log, wood, rock, boulder, leaf, green leaf, root, mud, water, pond, water puddle, water ripple, reflection, "
    "grass, dirt, ground plane, cast shadow, drop shadow, scenery, foliage, landscape, sky, "
    "multiple animals, duplicate, two heads, double headed, two tails, extra limbs, closed ring, donut loop, fused coils, impossible coils, "
    "tiling, text, watermark, border, cropped body, cut off, waterline cutoff"
)


def get_clean_feet() -> tuple[Image.Image, Image.Image, Image.Image]:
    teal_gen = Image.open(EVAL_DIR / "clean_v6_step1250_s076" / "03_green-winged_teal.png").convert("RGB")
    feet_crop = teal_gen.crop((450, 657, 560, 725))
    f_arr = np.array(feet_crop, dtype=np.float32)
    f_fg = np.any(f_arr < 242, axis=-1)
    f_lum = 0.299 * f_arr[:, :, 0] + 0.587 * f_arr[:, :, 1] + 0.114 * f_arr[:, :, 2]

    f_gray_rgb = np.stack([
        np.where(f_fg, np.clip(f_lum * 0.72 + 15, 25, 165), 255),
        np.where(f_fg, np.clip(f_lum * 0.75 + 18, 25, 170), 255),
        np.where(f_fg, np.clip(f_lum * 0.78 + 22, 28, 175), 255),
    ], axis=-1).astype(np.uint8)

    f_tan_rgb = np.stack([
        np.where(f_fg, np.clip(f_lum * 0.88 + 18, 28, 195), 255),
        np.where(f_fg, np.clip(f_lum * 0.78 + 15, 25, 175), 255),
        np.where(f_fg, np.clip(f_lum * 0.58 + 10, 20, 135), 255),
    ], axis=-1).astype(np.uint8)

    mask = Image.fromarray((f_fg * 255).astype(np.uint8))
    return Image.fromarray(f_gray_rgb), Image.fromarray(f_tan_rgb), mask


def enhance_bird_matte_and_feathers(arr: np.ndarray, fg: np.ndarray, y_head_cut: int = 440) -> np.ndarray:
    """1. Damps shiny 3D specular highlights and deep shadows into flat matte tones.
    2. Embosses VAE-scale (40x24px, ~5x3 latent pixels) overlapping feather scallops with 4.5px-thick
       charcoal-umber contour lines (-65 lum) and stepped gouache feather highlights (+22 lum) across
       body plumage so SDXL's 8x VAE downsampler preserves rich American-Robin-style feather texture!"""
    yy, xx = np.ogrid[:1024, :1024]
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lum_broad = gaussian_filter(lum, sigma=18.0)
    lum_fine = gaussian_filter(lum, sigma=2.2)
    detail = lum - lum_fine
    med_grad = lum_fine - lum_broad

    # Flatten broad 3D lighting gradients (0.38x) to eliminate glossy/shiny look
    flat_lum = lum_broad + med_grad * 0.38 + detail * 1.25
    flat_lum = np.clip(flat_lum, 40.0, 218.0)

    # VAE-scale overlapping feather scallops oriented along body (period 40px x 24px -> 5x3 latent pixels!)
    u = (xx * 0.94 + yy * 0.34) / 40.0
    v = (-xx * 0.34 + yy * 0.94) / 24.0
    v_row = np.floor(v)
    u_shifted = u + 0.5 * np.mod(v_row, 2.0)
    du = (u_shifted - np.floor(u_shifted)) - 0.5
    dv = v - v_row
    arc_curve = 0.20 + 1.35 * (du ** 2)
    arc_dist = np.abs(dv - arc_curve)
    # 4.5px-thick feather contour line (width 0.22 of 24px period) so 8x VAE downsampling sees it clearly!
    feather_line = np.exp(-((arc_dist / 0.22) ** 2))
    # Stepped gouache shading inside each feather scallop
    feather_step = np.where(dv > arc_curve, np.clip((dv - arc_curve) / 0.55, 0.0, 1.0), 0.0)

    dist = distance_transform_edt(fg)
    smoothness = np.clip((18.0 - np.abs(detail)) / 18.0, 0.35, 1.0)
    is_body_plumage = (dist > 8) & (yy > y_head_cut) & (flat_lum > 45) & (flat_lum < 212)
    feather_mod = (-62.0 * feather_line + 22.0 * feather_step) * smoothness
    flat_lum = np.where(is_body_plumage, np.clip(flat_lum + feather_mod, 32.0, 220.0), flat_lum)

    scale = flat_lum / np.maximum(lum, 1.0)
    out = np.where(fg[:, :, None], np.clip(arr * scale[:, :, None], 0, 255).astype(np.uint8), 255)
    return out


def prepare_inputs(in_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    feet_gray_im, feet_tan_im, feet_mask = get_clean_feet()
    yy, xx = np.ogrid[:1024, :1024]

    # 00: Northern Shoveler
    shov_im = Image.open(V5_EVAL / "inputs" / "00_northern_shoveler.png").convert("RGB")
    shov_arr = np.array(shov_im, dtype=np.float32)
    shov_fg = np.any(shov_arr < 248, axis=-1)
    shov_med = np.stack([gaussian_filter(shov_arr[:, :, c], sigma=4.5) for c in range(3)], axis=-1)
    w_belly_s = np.clip((xx - 300.0) / 30.0, 0.0, 1.0) * np.clip((570.0 - xx) / 30.0, 0.0, 1.0) * np.clip((yy - 535.0) / 30.0, 0.0, 1.0)
    shov_arr = shov_arr * (1.0 - 0.70 * w_belly_s[:, :, None]) + shov_med * (0.70 * w_belly_s[:, :, None])
    Image.fromarray(enhance_bird_matte_and_feathers(shov_arr, shov_fg, y_head_cut=480)).save(in_dir / "00_northern_shoveler.png")

    # 01: Northern Pintail (Option B: seamless rounded lower belly + slate-gray webbed feet + bold feather scallops)
    p_im = Image.open(V5_EVAL / "inputs" / "01_northern_pintail.png").convert("RGB")
    p_arr = np.array(p_im, dtype=np.float32)
    p_fg = np.any(p_arr < 248, axis=-1)
    belly_bot_p = 635.0 + 46.0 * np.sin(np.clip((xx - 285.0) / (715.0 - 285.0), 0.0, 1.0) * np.pi)
    belly_top_p = 595.0 + 20.0 * np.sin(np.clip((xx - 285.0) / (715.0 - 285.0), 0.0, 1.0) * np.pi)
    in_belly_p = (xx >= 285) & (xx <= 715) & (yy >= belly_top_p) & (yy <= belly_bot_p)
    w_p = np.clip((yy - belly_top_p) / np.maximum(belly_bot_p - belly_top_p, 1.0), 0.0, 1.0)
    belly_col_p = np.array([214.0, 209.0, 200.0])[None, None, :] * (1.0 - 0.14 * (w_p[:, :, None] ** 2))
    p_arr = np.where((in_belly_p & (~p_fg))[:, :, None], belly_col_p, p_arr)
    p_arr = np.where((in_belly_p & p_fg)[:, :, None], p_arr * (1.0 - (w_p[:, :, None] ** 1.6)) + belly_col_p * (w_p[:, :, None] ** 1.6), p_arr)
    rim_p = (yy >= belly_bot_p - 3.5) & (yy <= belly_bot_p) & (xx >= 292) & (xx <= 710)
    p_arr[rim_p] = [60.0, 55.0, 50.0]
    p_fg = p_fg | in_belly_p
    p_clean = Image.fromarray(enhance_bird_matte_and_feathers(p_arr, p_fg, y_head_cut=460))
    p_clean.paste(feet_gray_im, (465, 675), feet_mask)
    p_clean.save(in_dir / "01_northern_pintail.png")

    # 02: American Wigeon (Option B: seamless rounded white lower belly + slate-gray webbed feet + bold feather scallops)
    w_im = Image.open(V5_EVAL / "inputs" / "02_american_wigeon.png").convert("RGB")
    w_arr = np.array(w_im, dtype=np.float32)
    w_fg = np.any(w_arr < 248, axis=-1)
    belly_bot_w = 574.0 + 54.0 * np.sin(np.clip((xx - 305.0) / (762.0 - 305.0), 0.0, 1.0) * np.pi)
    belly_top_w = 552.0 + 14.0 * np.sin(np.clip((xx - 305.0) / (762.0 - 305.0), 0.0, 1.0) * np.pi)
    in_belly_w = (xx >= 305) & (xx <= 762) & (yy >= belly_top_w) & (yy <= belly_bot_w)
    w_w = np.clip((yy - belly_top_w) / np.maximum(belly_bot_w - belly_top_w, 1.0), 0.0, 1.0)
    belly_col_w = np.array([228.0, 224.0, 216.0])[None, None, :] * (1.0 - 0.12 * (w_w[:, :, None] ** 2))
    w_arr = np.where((in_belly_w & (~w_fg))[:, :, None], belly_col_w, w_arr)
    w_arr = np.where((in_belly_w & w_fg)[:, :, None], w_arr * (1.0 - (w_w[:, :, None] ** 1.5)) + belly_col_w * (w_w[:, :, None] ** 1.5), w_arr)
    rim_w = (yy >= belly_bot_w - 3.5) & (yy <= belly_bot_w) & (xx >= 312) & (xx <= 756)
    w_arr[rim_w] = [60.0, 55.0, 50.0]
    w_fg = w_fg | in_belly_w
    w_clean = Image.fromarray(enhance_bird_matte_and_feathers(w_arr, w_fg, y_head_cut=425))
    w_clean.paste(feet_gray_im, (490, 622), feet_mask)
    w_clean.save(in_dir / "02_american_wigeon.png")

    # 03: Green-winged Teal (NO blurring of gray flank—keep its natural gray flank + add bold feather scallops + clean feet!)
    t_im = Image.open(V5_EVAL / "inputs" / "03_green-winged_teal.png").convert("RGB")
    t_arr = np.array(t_im, dtype=np.float32)
    t_fg = np.any(t_arr < 248, axis=-1)
    t_clean = Image.fromarray(enhance_bird_matte_and_feathers(t_arr, t_fg, y_head_cut=415))
    t_clean.paste(feet_tan_im, (460, 625), feet_mask)
    t_clean.save(in_dir / "03_green-winged_teal.png")

    # 04: Common Watersnake (Water Ref 278839486 — Single-polygon seamless tapered tail point at (485, 592)!)
    s_water = Image.open(RAW_SHARP / "05_common_watersnake.png").convert("RGB")
    arr_w = np.array(s_water, dtype=np.float32)
    mask_w = Image.new("L", (1024, 1024), 0)
    d_w = ImageDraw.Draw(mask_w)
    d_w.polygon([
        (552, 462), (548, 425), (565, 390), (585, 368),
        (485, 358), (375, 368), (305, 405), (282, 460), (295, 515), (345, 558), (410, 582), (475, 596),
        (405, 614), (320, 620), (235, 592), (175, 540), (165, 475), (195, 390),
        (265, 320), (390, 280), (540, 280), (690, 305), (820, 355), (890, 425),
        (885, 475), (825, 498), (720, 492), (610, 482)
    ], fill=255)
    smooth_w = np.array(mask_w.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) > 128.0
    fg_sw = smooth_w & np.any(arr_w < 250, axis=-1)
    dist_sw = distance_transform_edt(fg_sw)
    lum_sw = 0.299 * arr_w[:, :, 0] + 0.587 * arr_w[:, :, 1] + 0.114 * arr_w[:, :, 2]
    lum_sw_loc = gaussian_filter(lum_sw, sigma=18.0)
    sub_w = np.clip((640.0 - xx) / 140.0, 0.0, 1.0) * np.clip((yy - 400.0) / 80.0, 0.0, 1.0)
    new_lum_sw = np.clip(lum_sw_loc + (lum_sw - lum_sw_loc) * (1.0 + 0.90 * sub_w), 28.0, 182.0)
    # Smoothly darken the last 110px of the tapered tail tip (x > 340, y > 515) so vertical crossbands at the tip
    # cannot be mistaken for labial jaw bars of a second head!
    tail_tip_w = np.clip((xx - 330.0) / 130.0, 0.0, 1.0) * (yy > 515).astype(np.float32) * (xx < 495).astype(np.float32)
    new_lum_sw = new_lum_sw * (1.0 - 0.85 * tail_tip_w) + 52.0 * (0.85 * tail_tip_w)
    is_tail_rim_w = fg_sw & (dist_sw <= 3.5) & (yy > 500) & (xx < 495)
    new_lum_sw = np.where(is_tail_rim_w, 26.0, new_lum_sw)
    out_sw = np.stack([
        np.where(fg_sw, np.clip(new_lum_sw * 1.06, 25, 205), 255),
        np.where(fg_sw, np.clip(new_lum_sw * 0.93, 22, 185), 255),
        np.where(fg_sw, np.clip(new_lum_sw * 0.78, 18, 160), 255),
    ], axis=-1).astype(np.uint8)
    Image.fromarray(out_sw).save(in_dir / "04_watersnake_278839486_taper.png")

    # 05: Common Watersnake (Dry Ref 15458841 — Single-polygon seamless tapered tail point at (415, 572)!)
    s_dry = Image.open(PRE_V6_DIR / "inputs" / "06_watersnake_15458841.png").convert("RGB")
    arr_d = np.array(s_dry, dtype=np.float32)
    mask_d = Image.new("L", (1024, 1024), 0)
    d_d = ImageDraw.Draw(mask_d)
    d_d.polygon([
        (480, 535), (492, 485), (565, 475), (620, 488), (628, 435),
        (530, 405), (370, 405), (268, 445), (262, 495), (305, 538), (360, 560), (412, 574),
        (345, 592), (245, 590), (150, 545), (120, 460), (160, 375), (280, 330),
        (470, 320), (645, 340), (725, 395), (728, 465), (685, 535), (635, 585),
        (555, 598), (498, 578)
    ], fill=255)
    smooth_d = np.array(mask_d.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) > 128.0
    fg_d = smooth_d & np.any(arr_d < 248, axis=-1)
    dist_d = distance_transform_edt(fg_d)
    lum_d = 0.299 * arr_d[:, :, 0] + 0.587 * arr_d[:, :, 1] + 0.114 * arr_d[:, :, 2]
    lum_d_loc = gaussian_filter(lum_d, sigma=14.0)
    new_lum_d = np.clip(lum_d_loc + (lum_d - lum_d_loc) * 2.2, 28.0, 192.0)
    # Smoothly darken the last 90px of the tapered tail tip (x in 310..420, y > 510) so it never splits at higher strength!
    tail_tip_d = np.clip((xx - 300.0) / 105.0, 0.0, 1.0) * (yy > 510).astype(np.float32) * (xx < 425).astype(np.float32)
    new_lum_d = new_lum_d * (1.0 - 0.85 * tail_tip_d) + 48.0 * (0.85 * tail_tip_d)
    is_tail_rim_d = fg_d & (dist_d <= 3.5) & (yy > 480) & (xx < 425)
    new_lum_d = np.where(is_tail_rim_d, 24.0, new_lum_d)
    scale_d = new_lum_d / np.maximum(lum_d, 1.0)
    Image.fromarray(np.where(fg_d[:, :, None], np.clip(arr_d * scale_d[:, :, None], 0, 255).astype(np.uint8), 255)).save(in_dir / "05_watersnake_15458841_taper.png")


def clean_post(src_path: Path, dst_path: Path) -> Image.Image:
    if dst_path.exists():
        return Image.open(dst_path).convert("RGB")
    im = Image.open(src_path).convert("RGBA")
    cut = remove(im)
    arr = np.array(cut)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    yy, xx = np.ogrid[:1024, :1024]
    is_ground_shadow = (yy > 640) & (xx < 435) & (r > 195) & (g > 195) & (b > 195)
    arr[is_ground_shadow | (a < 180), 3] = 0
    cut = Image.fromarray(arr)
    box = subject_bbox(cut)
    if box is None:
        rgb = im.convert("RGB")
        rgb.save(dst_path)
        return rgb
    subj = cut.crop(box)
    scale = int(1024 * 0.76) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj = subj.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj, ((1024 - nw) // 2, (1024 - nh) // 2), subj)
    rgb = canvas.convert("RGB")
    rgb.save(dst_path)
    return rgb


def run_sdxl(in_dir: Path, prompt_file: Path, out_dir: Path, clean_dir: Path, lora_path: Path, strength: str, cfg: str = "5.5", mul: str = "1.10") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    if len(list(clean_dir.glob("*.png"))) >= 6:
        print(f"  [cached] {clean_dir.name}", flush=True)
        return
    cmd = [
        str(PY), str(LORA / "sd-scripts" / "sdxl_gen_img.py"),
        "--ckpt", str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        "--outdir", str(out_dir),
        "--network_module", "networks.lora",
        "--network_weights", str(lora_path),
        "--network_mul", mul,
        "--max_embeddings_multiples", "3",
        "--from_file", str(prompt_file),
        "--image_path", str(in_dir),
        "--use_original_file_name",
        "--strength", strength,
        "--W", "1024", "--H", "1024",
        "--steps", "28", "--sampler", "euler_a",
        "--scale", cfg, "--bf16", "--sdpa",
        "--batch_size", "1", "--images_per_prompt", "1",
    ]
    subprocess.run(cmd, check=True)
    for png in sorted(out_dir.glob("*.png")):
        clean_post(png, clean_dir / png.name)


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = TEST_DIR / "inputs"
    prepare_inputs(in_dir)

    prompts = [
        # 00: Northern Shoveler
        f"animl_gouache_style, a northern shoveler, full body wildlife illustration, single bird in flight with orange feet visible, matte dark green head, yellow eye, black spatulate bill, crisp white breast, warm chestnut-orange belly and flanks with distinct overlapping feather contour lines, layered slate-gray wing feathers with pale blue wing patch, {STYLE_BIRD_V6} --n {NEG_V6}, spotted belly, scaled belly --d 42",
        # 01: Northern Pintail
        f"animl_gouache_style, a northern pintail, full body wildlife illustration, single standing duck with rounded white lower belly and gray webbed feet visible underneath, matte chocolate-brown head, white neck stripe and breast, bluish-gray bill, layered gray flank and back feathers with distinct overlapping feather contour lines, black-and-buff scapular feathers, pointed tail, {STYLE_BIRD_V6} --n {NEG_V6} --d 42",
        # 02: American Wigeon
        f"animl_gouache_style, an american wigeon, full body wildlife illustration, single standing duck with rounded white lower belly and gray webbed feet visible underneath, matte cream crown and green eye-patch on head, pale blue-gray bill with black tip, warm pinkish-brown breast and flanks with distinct overlapping feather contour lines, layered brown-and-gray wing feathers, black undertail border, {STYLE_BIRD_V6} --n {NEG_V6} --d 42",
        # 03: Green-winged Teal (keeps gray flank + layered feather contour lines!)
        f"animl_gouache_style, a green-winged teal, full body wildlife illustration, single standing duck with webbed feet visible underneath, matte chestnut-brown head with green eye-patch, vertical white shoulder stripe, dark slate bill, medium gray flanks and back with distinct overlapping feather contour lines, warm buff breast, green wing speculum, {STYLE_BIRD_V6} --n {NEG_V6}, stippled dots, speckled dots --d 42",
        # 04: Common Watersnake (Water Ref 278839486 Tapered Tail)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single C-shaped snake with one head on the right and a narrow pointed tail tip on the bottom-left, continuous unbranched body tapering smoothly to a sharp point at the tail tip, warm brown and tan scales with dark brown dorsal crossbands, pale labial scales with dark vertical bars, round pupil, {STYLE_SNAKE_V6} --n {NEG_V6} --d 42",
        # 05: Common Watersnake (Dry Ref 15458841 Tapered Tail)
        f"animl_gouache_style, a common watersnake, full body wildlife illustration, single C-shaped snake with one head on the right and a narrow pointed tail tip on the bottom-left, continuous unbranched body tapering smoothly to a sharp point at the tail tip, dark reddish-brown and gray-brown scales with dark dorsal crossbands, pale labial scales with dark bars, round pupil, {STYLE_SNAKE_V6} --n {NEG_V6} --d 42",
    ]
    pf = TEST_DIR / "prompts.txt"
    pf.write_text("\n".join(prompts), encoding="utf-8")

    lora_1250 = LORA / "output" / "animl_style_v6-step00001250.safetensors"

    run_sdxl(in_dir, pf, TEST_DIR / "raw_s071", TEST_DIR / "clean_s071", lora_1250, "0.71", "5.5", "1.10")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s074", TEST_DIR / "clean_s074", lora_1250, "0.74", "5.5", "1.10")
    run_sdxl(in_dir, pf, TEST_DIR / "raw_s077", TEST_DIR / "clean_s077", lora_1250, "0.77", "5.5", "1.10")

    rows = [
        ("00_northern_shoveler", "Northern Shoveler (Matte + Layered Feathers)", EVAL_DIR / "clean_v6_step1250_s076" / "00_northern_shoveler.png"),
        ("01_northern_pintail", "Northern Pintail (Option B Belly/Feet + Bold Feather Scallops)", EVAL_DIR / "clean_v6_step1250_s076" / "01_northern_pintail.png"),
        ("02_american_wigeon", "American Wigeon (Option B Belly/Feet + Bold Feather Scallops)", EVAL_DIR / "clean_v6_step1250_s076" / "02_american_wigeon.png"),
        ("03_green-winged_teal", "Green-winged Teal (Gray Flank + Layered Feather Scallops)", EVAL_DIR / "clean_v6_step1250_s076" / "03_green-winged_teal.png"),
        ("04_watersnake_278839486_taper", "Common Watersnake A (Water Photo 278839486 — Tapered Tail Tip)", EVAL_DIR / "clean_v6_step1250_s076" / "09_common_watersnake.png"),
        ("05_watersnake_15458841_taper", "Common Watersnake B (Dry Photo 15458841 — Tapered Tail Tip)", EVAL_DIR / "clean_v6_step1250_s076" / "09_common_watersnake.png"),
    ]

    CELL = 360
    HEADER_H = 50
    ROW_LABEL_H = 34
    W = 5 * CELL
    H = HEADER_H + len(rows) * (CELL + ROW_LABEL_H)
    sheet = Image.new("RGB", (W, H), (248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 17)
        font_r = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_b = ImageFont.load_default()
        font_r = font_b

    col_titles = [
        "1. Upgraded Input",
        "2. Previous v6 (s=0.76)",
        "3. New v6 (s=0.71)",
        "4. New v6 (s=0.74)",
        "5. New v6 (s=0.77)",
    ]
    for c_idx, title in enumerate(col_titles):
        draw.text((c_idx * CELL + 12, 15), title, fill=(20, 20, 30), font=font_b)

    for r_idx, (stem, lbl, prev_p) in enumerate(rows):
        y0 = HEADER_H + r_idx * (CELL + ROW_LABEL_H)
        draw.rectangle([0, y0, W, y0 + ROW_LABEL_H], fill=(232, 235, 242))
        draw.text((14, y0 + 8), lbl, fill=(30, 30, 50), font=font_r)
        paths = [
            in_dir / f"{stem}.png",
            prev_p,
            TEST_DIR / "clean_s071" / f"{stem}.png",
            TEST_DIR / "clean_s074" / f"{stem}.png",
            TEST_DIR / "clean_s077" / f"{stem}.png",
        ]
        for c_idx, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB").resize((CELL - 8, CELL - 8), Image.LANCZOS)
                sheet.paste(im, (c_idx * CELL + 4, y0 + ROW_LABEL_H + 4))
            draw.rectangle([c_idx * CELL, y0 + ROW_LABEL_H, (c_idx + 1) * CELL, y0 + ROW_LABEL_H + CELL], outline=(210, 215, 220), width=1)

    out_sheet = ARTIFACT_DIR / "v6_birds_and_snake_fixes_compare.jpg"
    sheet.save(out_sheet, quality=93)
    print(f"Saved {out_sheet}", flush=True)


if __name__ == "__main__":
    main()
