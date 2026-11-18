"""End-to-End Automated Pipeline Test on 20 Brand-New Unseen Expansion Species.

Stages (100% automated, zero manual masks or hand-picked URLs):
1. Multi-Tier Photo Fetching + CLIP + BiRefNet Cutout + Gemma3:27b Selector & Trait Extractor:
   - Fetches candidate photos from Wikimedia Commons, iNaturalist Curated Taxon Guide, and top-voted iNat Research-Grade observations.
   - Ranks candidates with OpenCLIP (`ViT-B-32`) for full-body side/3/4 view clarity.
   - Runs `birefnet-general` on the top 3 candidates per species, normalizes to 1024x1024, and uses `gemma3:27b` + Wikipedia `== Description ==` to pick the most complete/natural cutout and extract accurate species color/marking traits.
2. Automated 3D Depth + Contrast-Adaptive Bilateral Canny Conditioning:
   - Runs `Depth-Anything-V2-Small-hf` (unclipped min-max normalization) + contrast-adaptive Bilateral Canny.
3. Class-Calibrated SDXL + `animl_style_v6` + 225-Token Multi-Chunk Generation:
   - Uses `encode_prompt_sdxl_multichunk` (`max_chunks=3`, 225 tokens) so zero `STYLE_V6` or `NEG_V6` tokens are truncated.
   - Uses full `animl_style_v6-step00001250.safetensors` for Mammals, Birds, Lizards, Turtles, Amphibians, and Sharks/Rays, and `animl_style_v6_unet_only.safetensors` + `strip_head_eye_clauses` for Snakes.
"""

import base64
import json
import re
import shutil
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)

from fetch_photo_candidates import (
    fetch_wikimedia_commons,
    fetch_inat_curated_taxon,
    fetch_inat_obs,
)
from wiki_descriptions import fetch_wiki_batch, extract_visual_clauses
from u2net_matte import remove
from normalize import subject_bbox

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")
OLLAMA_URL = "http://localhost:11434"

# 20 brand-new unseen species across all 5 classes from expansion/master_species_list.json
FRESH_20_SPECIES = [
    # Mammals (5)
    ("01_desert_cottontail", "Desert Cottontail", "Sylvilagus audubonii", 43115, "Mammalia", "mammal"),
    ("02_black_tailed_jackrabbit", "Black-tailed Jackrabbit", "Lepus californicus", 43130, "Mammalia", "mammal"),
    ("03_golden_mantled_squirrel", "Common Golden-mantled Ground Squirrel", "Callospermophilus lateralis", 180010, "Mammalia", "mammal"),
    ("04_western_gray_squirrel", "Western Gray Squirrel", "Sciurus griseus", 46023, "Mammalia", "mammal"),
    ("05_yellow_bellied_marmot", "Yellow-bellied Marmot", "Marmota flaviventris", 46086, "Mammalia", "mammal"),
    # Birds (5)
    ("06_house_finch", "House Finch", "Haemorhous mexicanus", 199840, "Aves", "bird"),
    ("07_mourning_dove", "Mourning Dove", "Zenaida macroura", 3454, "Aves", "bird"),
    ("08_red_winged_blackbird", "Red-winged Blackbird", "Agelaius phoeniceus", 9744, "Aves", "bird"),
    ("09_california_scrub_jay", "California Scrub-Jay", "Aphelocoma californica", 506118, "Aves", "bird"),
    ("10_pied_billed_grebe", "Pied-billed Grebe", "Podilymbus podiceps", 4246, "Aves", "bird"),
    # Reptiles: Snakes, Lizards, Turtles (5)
    ("11_ring_necked_snake", "Ring-necked Snake", "Diadophis punctatus", 26575, "Reptilia", "snake"),
    ("12_dekays_brownsnake", "DeKay's Brownsnake", "Storeria dekayi", 28562, "Reptilia", "snake"),
    ("13_side_blotched_lizard", "Common Side-blotched Lizard", "Uta stansburiana", 36094, "Reptilia", "lizard"),
    ("14_southern_alligator_lizard", "Southern Alligator Lizard", "Elgaria multicarinata", 32055, "Reptilia", "lizard"),
    ("15_spiny_softshell", "Spiny Softshell", "Apalone spinifera", 39556, "Reptilia", "turtle"),
    # Amphibians: Frogs, Toads, Newts (3)
    ("16_green_treefrog", "Green Treefrog", "Dryophytes cinereus", 1668858, "Amphibia", "amphibian"),
    ("17_wood_frog", "Wood Frog", "Lithobates sylvaticus", 66012, "Amphibia", "amphibian"),
    ("18_california_newt", "California Newt", "Taricha torosa", 27818, "Amphibia", "amphibian"),
    # Sharks & Rays (2)
    ("19_leopard_shark", "Leopard Shark", "Triakis semifasciata", 52297, "Elasmobranchii", "shark_ray"),
    ("20_bat_ray", "Bat Ray", "Myliobatis californica", 51491, "Elasmobranchii", "shark_ray"),
]

# Exact 100% parity with evaluate_v6_suite.py
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
    "grass, dirt, ground plane, cast shadow, scenery, foliage, landscape, sky, human hand, fingers, person, "
    "multiple animals, duplicate, two tails, extra limbs, extra feet, extra paws, row of feet, claws on belly, fused legs, tiling, text, watermark, border, cropped body, cut off"
)


def download_pil(url: str, timeout: int = 25) -> Image.Image | None:
    try:
        r = requests.get(url, headers={"User-Agent": "AnimlVisionArtPipeline/1.0"}, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def make_normalized_cutout(im: Image.Image, subtype: str = "") -> tuple[Image.Image | None, Image.Image | None, dict]:
    """Run BiRefNet (`remove`) on `im`, enforce single-contiguous-subject geometry, and normalize to 1024x1024."""
    cut = remove(im.convert("RGBA"))
    arr = np.array(cut)
    alpha = arr[:, :, 3]
    fg_mask = (alpha >= 165).astype(np.uint8)
    fg_frac = float(fg_mask.mean())
    if fg_frac < 0.03 or fg_frac > 0.85:
        return None, None, {"valid": False, "reason": f"bad fg_frac={fg_frac:.3f}"}

    # Connected-component analysis: reject photos where a reed/branch split the animal into 2 pieces,
    # and strip any tiny detached background specks (< 1.2% of main subject area).
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
    if num_l <= 1:
        return None, None, {"valid": False, "reason": "no components"}
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    main_label = int(order[0]) + 1
    main_area = float(areas[order[0]])
    if len(order) > 1:
        sec_area = float(areas[order[1]])
        if sec_area / max(1.0, main_area) >= 0.015:
            return None, None, {"valid": False, "reason": f"split_subject (2nd comp={sec_area/main_area:.3f})"}

    # Keep strictly the single main contiguous animal component
    clean_mask = (labels == main_label)
    arr[~clean_mask, 3] = 0

    # Automatic underwater green/cyan color-cast neutralization for sharks/rays
    if subtype == "shark_ray":
        rgb_fg = arr[clean_mask, :3].astype(np.float32)
        r_m, g_m, b_m = rgb_fg[:, 0].mean(), rgb_fg[:, 1].mean(), rgb_fg[:, 2].mean()
        if (g_m - r_m) > 18.0 or (b_m - r_m) > 18.0:
            lab = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
            a_fg = lab[clean_mask, 1].mean()
            b_fg = lab[clean_mask, 2].mean()
            lab[:, :, 1] = np.clip(lab[:, :, 1] + (128.0 - a_fg) * 0.85, 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + (132.0 - b_fg) * 0.75, 0, 255)
            arr[:, :, :3] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Check how much of the subject is clipped by the original photo borders
    top_touch = float(clean_mask[0, :].mean())
    bot_touch = float(clean_mask[-1, :].mean())
    left_touch = float(clean_mask[:, 0].mean())
    right_touch = float(clean_mask[:, -1].mean())
    edge_touch_sum = top_touch + bot_touch + left_touch + right_touch
    if edge_touch_sum > 0.12:
        return None, None, {"valid": False, "reason": f"border_clipped ({edge_touch_sum:.3f})"}

    cut_clean = Image.fromarray(arr)
    box = subject_bbox(cut_clean)
    if box is None:
        return None, None, {"valid": False, "reason": "empty bbox"}

    subj = cut_clean.crop(box)
    scale = int(1024 * 0.78) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj_r = subj.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj_r, ((1024 - nw) // 2, (1024 - nh) // 2), subj_r)
    norm_rgb = canvas.convert("RGB")

    # Also build a 1024x1024 padded original photo for side-by-side comparison
    im_scale = min(1024 / im.width, 1024 / im.height)
    pw, ph = max(1, round(im.width * im_scale)), max(1, round(im.height * im_scale))
    photo_pad = Image.new("RGB", (1024, 1024), (245, 245, 245))
    photo_pad.paste(im.resize((pw, ph), Image.LANCZOS), ((1024 - pw) // 2, (1024 - ph) // 2))

    return norm_rgb, photo_pad, {
        "valid": True,
        "fg_frac": round(fg_frac, 4),
        "edge_touch_sum": round(edge_touch_sum, 4),
    }


def query_gemma27b_cutout_and_traits(
    cutout_im: Image.Image,
    common_name: str,
    sci_name: str,
    wiki_clauses: str,
) -> dict:
    """Ask Gemma3:27b to rate the cutout (0-10) for full-body field-guide suitability
    AND extract a concise comma-separated visual trait description grounded in Wikipedia."""
    buf = BytesIO()
    cutout_im.resize((512, 512), Image.LANCZOS).save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f"You are evaluating a segmented cutout of a {common_name} ({sci_name}) on a white background "
        f"for a wildlife field-guide illustration.\n"
        f"Wikipedia reference description for {common_name}: {wiki_clauses}\n\n"
        f"Return STRICT JSON only with these keys:\n"
        f"1. \"single_complete_animal\": boolean (true ONLY if the ENTIRE animal including head, body, tail, and legs/wings/fins is clearly visible and NOT cut in half by leaves/reeds or curled inside a dark hole)\n"
        f"2. \"no_fused_branch_or_rock\": boolean (true ONLY if no rock, hole rim, dead leaves, branch, or human hand is stuck to the animal cutout)\n"
        f"3. \"pose_quality_score\": number from 0 to 10 (10 = ideal clear side or 3/4 full-body field guide pose; <= 3 if half the body is missing, buried in leaves, or inside a hole)\n"
        f"4. \"facing_direction\": \"left\" or \"right\"\n"
        f"5. \"visual_traits\": string of 18 to 30 words listing the specific colors, markings, patterns, and anatomical features of this {common_name} (combining what is visible in the cutout with the Wikipedia reference description; comma-separated phrases, no background words)."
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "gemma3:27b",
                "stream": False,
                "keep_alive": "5m",
                "format": "json",
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            },
            timeout=120,
        )
        r.raise_for_status()
        data = json.loads(r.json()["message"]["content"])
        return data
    except Exception as e:
        print(f"  [Gemma warning for {common_name}]: {e}", flush=True)
        return {
            "single_complete_animal": True,
            "no_fused_branch_or_rock": True,
            "pose_quality_score": 6.0,
            "facing_direction": "right",
            "visual_traits": wiki_clauses[:180] if wiki_clauses else f"natural {common_name} coloration and markings",
        }


def unload_ollama_model(model_name: str = "gemma3:27b") -> None:
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass


def stage1_select_and_cutout(
    photo_dir: Path,
    cutout_dir: Path,
    meta_path: Path,
) -> dict:
    photo_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if len(meta) == len(FRESH_20_SPECIES) and all((cutout_dir / f"{s[0]}.png").exists() for s in FRESH_20_SPECIES):
            print("Stage 1 (Photo Selection + BiRefNet Cutouts + Gemma3:27b Traits) already cached.", flush=True)
            return meta
    else:
        meta = {}

    # 1. Fetch Wikipedia descriptions for all 20 species
    print("Fetching Wikipedia descriptions for all 20 species...", flush=True)
    wiki_titles = [s[1] for s in FRESH_20_SPECIES]
    wiki_raw = fetch_wiki_batch(wiki_titles)
    wiki_clauses_map = {}
    for _, common, sci, _, _, _ in FRESH_20_SPECIES:
        txt = wiki_raw.get(common, "")
        if not txt:
            txt = fetch_wiki_batch([sci]).get(sci, "")
        wiki_clauses_map[common] = extract_visual_clauses(txt)

    # 2. Load CLIP ViT-B-32 via transformers for rapid candidate photo pre-ranking
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(FRESH_20_SPECIES):
        out_cut = cutout_dir / f"{stem}.png"
        out_photo = photo_dir / f"{stem}.png"
        if stem in meta and out_cut.exists() and out_photo.exists():
            print(f"[Stage 1 {idx+1}/20] {stem} already cached.", flush=True)
            continue

        print(f"[Stage 1 {idx+1}/20] Fetching & ranking candidates for {common} ({sci})...", flush=True)
        cands = []
        seen_ids = set()
        for c in fetch_inat_curated_taxon(taxon_id, limit=8):
            if c["photo_id"] not in seen_ids:
                seen_ids.add(c["photo_id"])
                cands.append(c)
        for c in fetch_wikimedia_commons(sci, common, limit=6):
            if c["photo_id"] not in seen_ids:
                seen_ids.add(c["photo_id"])
                cands.append(c)
        try:
            for c in fetch_inat_obs(taxon_id, "votes", 10):
                if c["photo_id"] not in seen_ids:
                    seen_ids.add(c["photo_id"])
                    cands.append(c)
        except Exception:
            pass

        # Download medium thumbnails and score with CLIP
        pos_prompt = f"a sharp clear full body side view photograph of a single adult {common}, whole animal visible"
        neg_prompts = [
            "a drawing, painting, map, or illustration",
            "an animal held in a human hand or fingers",
            "a dead, crushed, or skeletal animal",
            "an extreme close-up of only a face or eye",
            "a distant tiny blurry animal hidden in branches",
        ]
        all_texts = [pos_prompt] + neg_prompts

        scored_cands = []
        for c in cands[:18]:
            im_med = download_pil(c["medium_url"], timeout=15)
            if im_med is None:
                continue
            w, h = im_med.size
            if min(w, h) < 220 or max(w / h, h / w) > 2.6:
                continue
            with torch.no_grad():
                inputs = clip_proc(text=all_texts, images=im_med, return_tensors="pt", padding=True).to(device)
                outputs = clip_model(**inputs)
                img_f = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                txt_f = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                sims = (img_f @ txt_f.T).squeeze(0).float().cpu().numpy()
            clip_score = float(sims[0] - sims[1:].max())
            # Small bonus for curated taxon guide photos
            tier_bonus = 0.025 if c.get("source_tier") == "inat_curated_taxon" else 0.0
            scored_cands.append((clip_score + tier_bonus, c))

        scored_cands.sort(key=lambda x: x[0], reverse=True)

        # Evaluate top 8 candidates with BiRefNet single-component cutout + Gemma3:27b
        best_pack = None
        best_total_score = -999.0
        valid_checked = 0

        for rank_i, (c_score, c_info) in enumerate(scored_cands[:10]):
            if valid_checked >= 4 and best_total_score >= 16.5:
                break
            im_orig = download_pil(c_info["original_url"], timeout=25) or download_pil(c_info["medium_url"], timeout=20)
            if im_orig is None:
                continue
            norm_cut, pad_photo, cut_stats = make_normalized_cutout(im_orig, subtype=subtype)
            if not cut_stats.get("valid") or norm_cut is None:
                print(f"  cand #{rank_i+1} (id={c_info['photo_id']}): rejected by cutout geometry ({cut_stats.get('reason')})", flush=True)
                continue
            valid_checked += 1

            gemma_res = query_gemma27b_cutout_and_traits(
                norm_cut, common, sci, wiki_clauses_map.get(common, "")
            )
            pose_q = float(gemma_res.get("pose_quality_score", 5.0))
            complete_bonus = 4.0 if gemma_res.get("single_complete_animal", True) else -6.0
            clean_bonus = 2.5 if gemma_res.get("no_fused_branch_or_rock", True) else -3.5
            edge_penalty = float(cut_stats.get("edge_touch_sum", 0.0)) * 8.0
            total_s = c_score * 20.0 + pose_q + complete_bonus + clean_bonus - edge_penalty

            print(
                f"  cand #{rank_i+1} (id={c_info['photo_id']}): clip={c_score:.3f}, "
                f"pose={pose_q:.1f}, complete={gemma_res.get('single_complete_animal')}, "
                f"clean={gemma_res.get('no_fused_branch_or_rock')} -> total={total_s:.2f}",
                flush=True,
            )
            if total_s > best_total_score:
                best_total_score = total_s
                best_pack = (norm_cut, pad_photo, c_info, gemma_res, cut_stats, total_s)

        if best_pack is None:
            raise RuntimeError(f"No valid photo candidate found for {common}")

        norm_cut, pad_photo, c_info, gemma_res, cut_stats, total_s = best_pack
        norm_cut.save(out_cut)
        pad_photo.save(out_photo)
        meta[stem] = {
            "stem": stem,
            "common_name": common,
            "scientific_name": sci,
            "taxon_id": taxon_id,
            "class": cls_name,
            "subtype": subtype,
            "photo_id": c_info["photo_id"],
            "source_tier": c_info.get("source_tier"),
            "observation_url": c_info.get("observation_url"),
            "attribution": c_info.get("attribution"),
            "license": c_info.get("license"),
            "wiki_clauses": wiki_clauses_map.get(common, ""),
            "visual_traits": gemma_res.get("visual_traits", ""),
            "facing_direction": gemma_res.get("facing_direction", "right"),
            "pose_quality_score": gemma_res.get("pose_quality_score", 7.0),
            "total_score": round(total_s, 2),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Free VRAM from OpenCLIP and Ollama Gemma3:27b
    del clip_model
    unload_ollama_model("gemma3:27b")
    torch.cuda.empty_cache()
    print("Unloaded OpenCLIP and Gemma3:27b from VRAM.", flush=True)
    return meta


def stage2_build_conditions(cutout_dir: Path, depth_dir: Path, canny_dir: Path) -> None:
    depth_dir.mkdir(parents=True, exist_ok=True)
    canny_dir.mkdir(parents=True, exist_ok=True)

    if all((depth_dir / f"{s[0]}.png").exists() and (canny_dir / f"{s[0]}.png").exists() for s in FRESH_20_SPECIES):
        print("Stage 2 (Depth-Anything-V2 + Contrast-Adaptive Bilateral Canny) already cached.", flush=True)
        return

    print("Running Depth-Anything-V2-Small-hf + Contrast-Adaptive Bilateral Canny on all 20 cutouts...", flush=True)
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to("cuda")

    for stem, _, _, _, _, _ in FRESH_20_SPECIES:
        if (depth_dir / f"{stem}.png").exists() and (canny_dir / f"{stem}.png").exists():
            continue
        im = Image.open(cutout_dir / f"{stem}.png").convert("RGB")
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

        # Unclipped min-max foreground depth normalization
        d_fg = depth[fg]
        d_min, d_max = float(d_fg.min()), float(d_fg.max())
        d_norm = np.zeros_like(depth, dtype=np.uint8)
        d_scaled = (depth - d_min) / max(1e-5, d_max - d_min)
        d_norm[fg] = (45 + d_scaled[fg] * 210).astype(np.uint8)
        Image.fromarray(np.stack([d_norm] * 3, axis=-1)).save(depth_dir / f"{stem}.png")

        # 3D occlusion boundaries + contrast-adaptive Bilateral Canny
        depth_inp = cv2.inpaint(d_norm, mask_inv, 5, cv2.INPAINT_TELEA)
        depth_smooth = cv2.GaussianBlur(depth_inp, (9, 9), 2.0)
        depth_edges = cv2.Canny(depth_smooth, 24, 62)
        depth_edges = cv2.bitwise_and(depth_edges, fg_inner)
        depth_edges_strong = cv2.dilate(depth_edges, np.ones((2, 2), np.uint8), iterations=1)

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
        combined = cv2.bitwise_or(depth_edges_strong, cv2.bitwise_or(rgb_clean, sil_edges))
        Image.fromarray(np.stack([combined] * 3, axis=-1)).save(canny_dir / f"{stem}.png")

    del depth_model
    del processor
    torch.cuda.empty_cache()
    print("Unloaded Depth-Anything-V2 from VRAM.", flush=True)


def encode_prompt_sdxl_multichunk(
    pipe: StableDiffusionXLControlNetImg2ImgPipeline,
    prompt: str,
    negative_prompt: str,
    max_chunks: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """225-token Multi-Chunk SDXL Prompt Encoder (zero 77-token truncation)."""
    device = pipe._execution_device
    dtype = pipe.unet.dtype

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
    clauses = [c.strip() for c in raw_traits.split(",") if c.strip()]
    kept = []
    for c in clauses:
        low = c.lower()
        if any(w in low for w in ("pupil", "eye", "head", "labial")):
            continue
        kept.append(c)
    return ", ".join(kept)


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
    photo_dir = WORK / "photos_selected"
    cutout_dir = WORK / "cutouts_1024"
    depth_dir = WORK / "depth"
    canny_dir = WORK / "canny"
    raw_dir = WORK / "raw_out"
    clean_dir = WORK / "clean_out"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    meta_path = WORK / "fresh_20_metadata.json"

    # Stage 1: Photo selection + BiRefNet Cutout + Gemma3:27b + Wikipedia traits
    meta = stage1_select_and_cutout(photo_dir, cutout_dir, meta_path)

    # Stage 2: Depth-Anything-V2 + Contrast-Adaptive Bilateral Canny
    stage2_build_conditions(cutout_dir, depth_dir, canny_dir)

    # Stage 3: Load Dual ControlNet + SDXL pipeline
    print("Loading Dual ControlNet + SDXL pipeline...", flush=True)
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

    # Pass A: Run the 18 non-snake species with FULL animl_style_v6-step00001250.safetensors
    non_snakes = [s for s in FRESH_20_SPECIES if s[5] != "snake"]
    snakes = [s for s in FRESH_20_SPECIES if s[5] == "snake"]

    if any(not (clean_dir / f"{s[0]}.png").exists() for s in non_snakes):
        print("Loading FULL animl_style_v6-step00001250.safetensors for non-snake species...", flush=True)
        pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6-step00001250.safetensors")
        pipe.fuse_lora(lora_scale=1.0)

        for stem, common, sci, taxon_id, cls_name, subtype in non_snakes:
            out_raw = raw_dir / f"{stem}.png"
            out_clean = clean_dir / f"{stem}.png"
            if out_clean.exists():
                continue

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

            # Class-calibrated strength & ControlNet conditioning
            if subtype == "amphibian":
                strength, c_scales, c_ends = 0.55, [0.46, 0.28], [0.76, 0.60]
            elif subtype in ("lizard", "turtle", "shark_ray"):
                strength, c_scales, c_ends = 0.58, [0.48, 0.30], [0.78, 0.62]
            else:  # mammal, bird
                strength, c_scales, c_ends = 0.58, [0.42, 0.26], [0.74, 0.58]

            print(f"[Stage 3 Non-Snake] {stem} ({common}, s={strength}, scales={c_scales})...", flush=True)
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
            res.save(out_raw)
            clean_post(res, out_clean)

        pipe.unfuse_lora()
        pipe.unload_lora_weights()

    # Pass B: Run the 2 snake species with animl_style_v6_unet_only.safetensors + strip_head_eye_clauses
    if any(not (clean_dir / f"{s[0]}.png").exists() for s in snakes):
        print("Loading animl_style_v6_unet_only.safetensors for snake species...", flush=True)
        pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6_unet_only.safetensors")
        pipe.fuse_lora(lora_scale=1.05)

        for stem, common, sci, taxon_id, cls_name, subtype in snakes:
            out_raw = raw_dir / f"{stem}.png"
            out_clean = clean_dir / f"{stem}.png"
            if out_clean.exists():
                continue

            init_im = Image.open(cutout_dir / f"{stem}.png").convert("RGB")
            canny_im = Image.open(canny_dir / f"{stem}.png").convert("RGB")
            depth_im = Image.open(depth_dir / f"{stem}.png").convert("RGB")

            do_mirror = str(meta[stem].get("facing_direction", "right")).lower().startswith("left")
            if do_mirror:
                init_im = ImageOps.mirror(init_im)
                canny_im = ImageOps.mirror(canny_im)
                depth_im = ImageOps.mirror(depth_im)

            raw_traits = meta[stem].get("visual_traits") or meta[stem].get("wiki_clauses") or f"natural {common} coloration"
            body_traits = strip_head_eye_clauses(raw_traits)
            prompt = (
                f"animl_gouache_style, field guide illustration of a {common}, "
                f"{body_traits}, {STYLE_V6}"
            )
            pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_sdxl_multichunk(
                pipe, prompt, NEG_V6_UNIFIED, max_chunks=3
            )

            print(f"[Stage 3 Snake] {stem} ({common}, mirror={do_mirror})...", flush=True)
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
            clean_post(res, out_clean)

    # Copy individual outputs to artifact dir and build 2 comparison sheets (1..10 and 11..20) for crisp legibility
    for stem, _, _, _, _, _ in FRESH_20_SPECIES:
        shutil.copyfile(clean_dir / f"{stem}.png", ARTIFACT_DIR / f"fresh20_{stem}.png")

    try:
        f_hdr = ImageFont.truetype("arialbd.ttf", 16)
        f_sub = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        f_hdr = ImageFont.load_default()
        f_sub = f_hdr

    for part_idx, chunk in enumerate([FRESH_20_SPECIES[:10], FRESH_20_SPECIES[10:]], start=1):
        col_w = 340
        row_h = 370
        header_h = 54
        cols = 4
        rows = len(chunk)
        sheet = Image.new("RGB", (col_w * cols, header_h + row_h * rows), (246, 246, 246))
        draw = ImageDraw.Draw(sheet)

        headers = [
            "1. Auto-Selected Reference Photo",
            "2. Normalized BiRefNet Cutout",
            "3. 3D Depth + Adaptive Canny",
            "4. Automated animl_style_v6 Output",
        ]
        for c_idx, h_text in enumerate(headers):
            draw.text((c_idx * col_w + 12, 16), h_text, fill=(20, 20, 20), font=f_hdr)

        for r_idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(chunk):
            y0 = header_h + r_idx * row_h
            draw.line([(0, y0), (col_w * cols, y0)], fill=(210, 210, 210), width=2)
            imgs = [
                Image.open(photo_dir / f"{stem}.png").convert("RGB"),
                Image.open(cutout_dir / f"{stem}.png").convert("RGB"),
                Image.open(canny_dir / f"{stem}.png").convert("RGB"),
                Image.open(clean_dir / f"{stem}.png").convert("RGB"),
            ]
            for c_idx, im in enumerate(imgs):
                x0 = c_idx * col_w
                thumb = im.resize((col_w - 16, col_w - 16), Image.LANCZOS)
                sheet.paste(thumb, (x0 + 8, y0 + 26))
            m = meta.get(stem, {})
            label = f"{stem[:2]}. {common} ({sci}) [{cls_name}/{subtype}] — photo #{m.get('photo_id')} (score={m.get('total_score')})"
            draw.text((12, y0 + 5), label, fill=(10, 50, 110), font=f_sub)

        sheet_out = ARTIFACT_DIR / f"v6_fresh_20_e2e_part{part_idx}.jpg"
        sheet.save(sheet_out, quality=93)
        print(f"Saved {sheet_out}", flush=True)


if __name__ == "__main__":
    main()
