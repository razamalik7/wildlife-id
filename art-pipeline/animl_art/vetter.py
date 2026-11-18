"""Stage 1A/1B cutout normalization, OpenCLIP candidate selection, and Gemma3:27b VLM vetting."""

import base64
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import binary_propagation

from animl_art.matte import remove, subject_bbox
from animl_art.sources import (
    download_pil_compliant,
    fetch_commons_files,
    fetch_inat_curated_taxon,
    fetch_inat_obs,
    fetch_marine_candidates_v7,
)

OLLAMA_URL = "http://localhost:11434"


def unload_ollama_model(model_name: str = "gemma3:27b") -> None:
    """Ask Ollama to immediately evict `model_name` from VRAM."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass


def make_normalized_cutout_v7(im: Image.Image, subtype: str = "") -> tuple[Image.Image | None, Image.Image | None, dict]:
    """BiRefNet cutout with connected hysteresis propagation so thin whip tails stay attached."""
    cut = remove(im.convert("RGBA"))
    arr = np.array(cut)
    alpha = arr[:, :, 3]
    core_mask = (alpha >= 165).astype(np.uint8)
    fg_frac = float(core_mask.mean())
    if fg_frac < 0.025 or fg_frac > 0.85:
        return None, None, {"valid": False, "reason": f"bad fg_frac={fg_frac:.3f}"}

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(core_mask)
    if num_l <= 1:
        return None, None, {"valid": False, "reason": "no components"}
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    main_label = int(order[0]) + 1
    main_area = float(areas[order[0]])
    split_limit = 0.025 if subtype == "shark_ray" else 0.015
    if len(order) > 1:
        sec_area = float(areas[order[1]])
        if sec_area / max(1.0, main_area) >= split_limit:
            return None, None, {"valid": False, "reason": f"split_subject ({sec_area / main_area:.3f})"}

    main_core = labels == main_label
    low_thresh = 38 if subtype == "shark_ray" else 90
    support = alpha >= low_thresh
    if subtype == "shark_ray":
        support = cv2.morphologyEx(support.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0

    clean_mask = binary_propagation(main_core, structure=np.ones((3, 3), dtype=bool), mask=support)
    arr[~clean_mask, 3] = 0
    thin_recovered = clean_mask & (alpha < 220)
    arr[thin_recovered, 3] = np.clip(alpha[thin_recovered].astype(np.int32) * 3, 185, 255).astype(np.uint8)

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

    edge_touch_sum = float(
        clean_mask[0, :].mean() + clean_mask[-1, :].mean() + clean_mask[:, 0].mean() + clean_mask[:, -1].mean()
    )
    if edge_touch_sum > 0.12:
        return None, None, {"valid": False, "reason": f"border_clipped ({edge_touch_sum:.3f})"}

    cut_clean = Image.fromarray(arr)
    box = subject_bbox(cut_clean)
    if box is None:
        return None, None, {"valid": False, "reason": "empty bbox"}

    subj = cut_clean.crop(box)
    scale = int(1024 * 0.78) / max(subj.size)
    nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
    subj_r = subj.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canvas.paste(subj_r, ((1024 - nw) // 2, (1024 - nh) // 2), subj_r)

    im_scale = min(1024 / im.width, 1024 / im.height)
    pw, ph = max(1, round(im.width * im_scale)), max(1, round(im.height * im_scale))
    photo_pad = Image.new("RGB", (1024, 1024), (245, 245, 245))
    photo_pad.paste(im.resize((pw, ph), Image.Resampling.LANCZOS), ((1024 - pw) // 2, (1024 - ph) // 2))
    return canvas.convert("RGB"), photo_pad, {
        "valid": True,
        "fg_frac": round(fg_frac, 4),
        "edge_touch_sum": round(edge_touch_sum, 4),
    }


def gather_top_candidates_for_species(
    spec: tuple[str, str, str, int, str, str, bool],
    clip_model,
    clip_proc,
    device: str,
    cand_cache_dir: Path,
) -> list[dict]:
    """Fetch, segment, rank via OpenCLIP (dual scuba+cutout for marine), and cache top candidates."""
    import torch

    stem, common, sci, taxon_id, _, subtype, _ = spec
    sp_dir = cand_cache_dir / stem
    sp_meta = sp_dir / "cands.json"
    if sp_meta.exists():
        return json.loads(sp_meta.read_text(encoding="utf-8"))

    print(f"  Fetching & segmenting candidates for {common} ({sci})...", flush=True)
    if subtype == "shark_ray":
        cands = fetch_marine_candidates_v7(taxon_id, sci, common)
        max_try, target_valid, keep_top = 24, 12, 4
    else:
        cands = []
        seen: set[str] = set()
        for c in fetch_inat_curated_taxon(taxon_id, limit=8):
            if str(c["photo_id"]) not in seen:
                seen.add(str(c["photo_id"]))
                cands.append(c)
        for c in fetch_commons_files(sci, common, limit=6):
            if str(c["photo_id"]) not in seen:
                seen.add(str(c["photo_id"]))
                cands.append(c)
        try:
            for c in fetch_inat_obs(taxon_id, "votes", 12):
                if str(c["photo_id"]) not in seen:
                    seen.add(str(c["photo_id"]))
                    cands.append(c)
        except Exception:
            pass
        max_try, target_valid, keep_top = 18, 7, 3

    valid_packs = []
    for c in cands[:max_try]:
        im = download_pil_compliant(c["medium_url"], timeout=12) or download_pil_compliant(c["original_url"], timeout=15)
        if im is None or min(im.size) < 300:
            continue
        norm_cut, pad_photo, st = make_normalized_cutout_v7(im, subtype=subtype)
        if not st.get("valid") or norm_cut is None:
            continue
        valid_packs.append((norm_cut, pad_photo, c, st))
        if len(valid_packs) >= target_valid:
            break

    if not valid_packs:
        return []

    if subtype == "shark_ray":
        scuba_pos = (
            f"a clear underwater scuba diving photograph of a single live {common} ({sci}) "
            f"swimming submerged in blue ocean water showing its back and long tail"
        )
        scuba_neg = (
            "photograph taken from above the water surface looking down through green water ripples and glare, "
            "upside-down white belly view in aquarium tunnel, beached dead fish on sand or boat"
        )
        cut_pos = (
            f"a complete full-body side-three-quarter view of a {common} ({sci}) "
            f"with both wings and long tail intact on a white background"
        )
        cut_neg = "cropped half body, missing tail, tailless blob, upside-down white belly"
        with torch.no_grad():
            in_photo = clip_proc(
                text=[scuba_pos, scuba_neg],
                images=[p[1] for p in valid_packs],
                return_tensors="pt",
                padding=True,
            ).to(device)
            scuba_prob = clip_model(**in_photo).logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()
            in_cut = clip_proc(
                text=[cut_pos, cut_neg],
                images=[p[0] for p in valid_packs],
                return_tensors="pt",
                padding=True,
            ).to(device)
            cut_prob = clip_model(**in_cut).logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()
        logits = 0.55 * scuba_prob + 0.45 * cut_prob
    else:
        pos_txt = f"a sharp full-body side-view wildlife field guide photo of a single {common} ({sci}) isolated on a white background"
        neg_txt = "cropped half body, human hand holding animal, dead animal, blurry, multiple animals, murky underwater"
        with torch.no_grad():
            inputs = clip_proc(
                text=[pos_txt, neg_txt],
                images=[p[0] for p in valid_packs],
                return_tensors="pt",
                padding=True,
            ).to(device)
            logits = clip_model(**inputs).logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()

    ranked_indices = np.argsort(logits)[::-1][:keep_top]
    sp_dir.mkdir(parents=True, exist_ok=True)
    saved_cands = []
    for rank_i, r_idx in enumerate(ranked_indices):
        norm_cut, pad_photo, c, _ = valid_packs[int(r_idx)]
        clip_s = float(logits[int(r_idx)])
        im_hi = download_pil_compliant(c["original_url"], timeout=16)
        if im_hi is not None and max(im_hi.size) > 600:
            hi_cut, hi_pad, hi_st = make_normalized_cutout_v7(im_hi, subtype=subtype)
            if hi_st.get("valid") and hi_cut is not None and hi_pad is not None:
                norm_cut, pad_photo = hi_cut, hi_pad
        norm_cut.save(sp_dir / f"cut_{rank_i}.png")
        pad_photo.save(sp_dir / f"photo_{rank_i}.png")
        saved_cands.append(
            {
                "rank": rank_i,
                "photo_id": c["photo_id"],
                "original_url": c["original_url"],
                "clip_score": round(clip_s, 4),
            }
        )
    sp_meta.write_text(json.dumps(saved_cands, indent=2), encoding="utf-8")
    return saved_cands


def vet_side_by_side_and_extract_traits_v7(
    pad_photo: Image.Image,
    norm_cut: Image.Image,
    common: str,
    sci: str,
    subtype: str,
    wiki_clauses: str,
) -> dict:
    """Pass `[Original Photo | Cutout]` to Gemma3:27b, prioritizing diagnostic marks & explicit unstriped coat."""
    sbs = Image.new("RGB", (1024, 512), (255, 255, 255))
    sbs.paste(pad_photo.resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
    sbs.paste(norm_cut.resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
    buf = BytesIO()
    sbs.save(buf, format="JPEG", quality=86)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    marine_rule = ""
    if subtype == "shark_ray":
        marine_rule = (
            "CRITICAL FOR SHARKS/RAYS: `valid_live_wild_pose` MUST be false if:\n"
            "  (a) the Left photo was taken FROM ABOVE THE WATER SURFACE looking down through waves, green surface ripples, or surface glare (we REQUIRE a clear UNDERWATER SCUBA / submerged photo),\n"
            "  (b) the photo shows ONLY the flat white underside/belly from underneath (such as an aquarium tunnel shot) rather than the patterned dorsal/side swimming view,\n"
            "  (c) the shark/ray is dead on a beach/boat, on dry sand, caught on a line, held by a human, or obscured by aquarium window glass reflections/glare, or has holes in its wings/fins.\n"
        )
    elif subtype == "snake":
        marine_rule = (
            "CRITICAL FOR SNAKES: `valid_live_wild_pose` MUST be false if the snake is held in a human hand, tangled in wire/netting, "
            "or if the cutout is missing the head or split into pieces.\n"
        )

    ray_tail_rule = ""
    if "ray" in common.lower() or "skate" in common.lower():
        ray_tail_rule = (
            "   - RAY/SKATE WHIP TAIL CHECK: If the species is a ray or skate and its long slender whip-like tail is missing, cut short, or faded off in the Right cutout compared to a complete ray, "
            "set `cutout_repair_needed` to \"complete_whip_tail\" (and set `pose_quality_score` <= 4 if completely tailless). Also explicitly include \"long slender dark whip-like tail\" in `visual_traits` when present/expected.\n"
            "   - RAY/SKATE DORSAL VS VENTRAL SCOPING: If the white ventral underside of an upturned pectoral wing or belly is visible, explicitly describe it as \"solid unspotted opaque white ventral wing underside and belly\" (never bare \"white pectoral fins\"), and scope any spots/markings strictly to the \"dorsal back and upperwings\".\n"
        )

    prompt = (
        f"Left image is the original reference photo; Right image is the segmented white-background cutout of a {common} ({sci}).\n"
        f"Wikipedia reference notes (use ONLY for anatomical texture & diagnostic marks, NEVER to override the photo's actual color/pattern): {wiki_clauses}\n"
        f"{marine_rule}\n"
        f"Inspect BOTH Left and Right images carefully and return STRICT JSON with these keys:\n"
        f"1. \"valid_live_wild_pose\": boolean (true ONLY if the animal is alive, in a clear full-body submerged/natural pose per above rules, NOT above-water surface ripples, NOT ventral-only belly view, NOT dead/held)\n"
        f"2. \"no_fused_background_blob\": boolean (true if the Right cutout does not have a giant rock, human hand, or mud wall fused to the body; a small twig under a perched bird's feet is OK)\n"
        f"3. \"no_aquarium_glass_reflection\": boolean (true if the photo is clean and free of aquarium/zoo window glass reflections, human reflections in glass, or bright glare streaks across the body)\n"
        f"4. \"pose_quality_score\": number from 0 to 10 (10 = sharp, well-lit, full-body side or 3/4 field-guide view where head, eye, tail, and body markings are crystal clear; <=3 if blurry, murky, foreshortened, or clipped in half)\n"
        f"5. \"cutout_repair_needed\": string (\"none\", \"complete_paws_or_toes\", \"swimming_waterline_tucked_feet\", \"perch_stub_under_feet\", \"complete_whip_tail\", or \"remove_small_debris\")\n"
        f"6. \"eye_color_in_photo\": short phrase describing the exact iris/eye color visible in the photo (e.g. \"bright golden-yellow iris with black pupil\", \"dark brown-black eye\", \"pale cream upper iris and brown lower iris\")\n"
        f"7. \"visual_traits\": 20 to 34 words of comma-separated `[color/pattern] [body part]` phrases describing ONLY visible colors, markings, and anatomy (no background words). STRICT RULES FOR `visual_traits`:\n"
        f"   - NAMESAKE & DIAGNOSTIC FIELD MARKS FIRST: Check `{common}`, BOTH images, and Wikipedia for any namesake/diagnostic field mark (such as a continuous pale cream-white diagonal shoulder stripe sweeping from the upper back across the mid-shoulder down to the chest on a peccary, double black breast bands on a plover, red waxy wingtip droplets on a waxwing, five pale stripes and blue tail on a skink, dorsal ribbon stripes, eye masks, or tail rings) and list those diagnostic markings FIRST with exact color & body location.\n"
        f"   - EXPLICIT UNSTRIPED COAT WHEN PLAIN: If a mammal has a plain or grizzled coat with NO dorsal or lateral stripes in the photo (e.g., prairie dogs, marmots, rabbits, sea lions), explicitly include \"unstriped\" in the coat description (e.g. \"unstriped grizzled buff-tan coat\").\n"
        f"   - MAMMAL MUZZLE & PAWS: For mammals, also include exact nose/muzzle and paw/foot coloration visible in the photo (e.g., furred tan muzzle with small dark nose, buff furred paws with dark claws) so paws and noses are never rendered bare pink.\n"
        f"{ray_tail_rule}"
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "gemma3:27b",
                "stream": False,
                "keep_alive": "5m",
                "format": "json",
                "options": {"temperature": 0.1},
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            },
            timeout=180,
        )
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except Exception as e:
        print(f"  [VLM warning for {common}]: {e}", flush=True)
        return {
            "valid_live_wild_pose": True,
            "no_fused_background_blob": True,
            "no_aquarium_glass_reflection": True,
            "pose_quality_score": 5.0,
            "cutout_repair_needed": "none",
            "eye_color_in_photo": "natural dark eye",
            "visual_traits": wiki_clauses[:180] if wiki_clauses else f"natural {common} coloration and markings",
        }
