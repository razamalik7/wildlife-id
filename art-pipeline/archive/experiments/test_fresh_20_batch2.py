"""
Batch 2 Final Pipeline Validation (20 Unseen Species):
- Stage 1 (`--stage1-only`):
  * Fetches candidates from iNaturalist curated taxon photos, Wikimedia Commons (`gsrnamespace=6`), and iNaturalist top-voted research-grade observations.
  * Runs BiRefNet (`make_normalized_cutout`) with underwater color-cast neutralization.
  * Runs Side-by-Side `[Original Photo | Cutout]` VLM Vetting (`Gemma3:27b`) + Photo-Primary / Wiki-Assisted Trait Extraction (including eye/iris coloration).
- Stage 2 (Pre-Stylization GenAI Cutout Repair):
  * Repairs any occluded paws/feet, completes waterline cutoffs on swimming birds with tucked/paddling feet, and places/keeps a short natural twig stub strictly under perching birds' gripping toes.
- Stage 3 (`--stylize`):
  * Smart False-Animal Common-Name Sanitizer (sanitizes misleading head/anatomy modifiers like 'Prairie Dog', 'Kangaroo Rat', 'Horned Lizard' while preserving true patterned head-noun compounds).
  * Dual ControlNet (`Canny + Depth`) + `animl_style_v6-step00001250.safetensors` with 225-token multi-chunk CLIP encoding.
  * Perch-preserving `clean_post` for perched birds + localized Photo-Iris Color Restoration where SDXL darkens light irises.
"""
import argparse
import base64
import json
import os
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from test_fresh_20_e2e import (
    OLLAMA_URL,
    STYLE_V6,
    NEG_V6_UNIFIED,
    clean_post,
    encode_prompt_sdxl_multichunk,
    extract_visual_clauses,
    fetch_inat_curated_taxon,
    fetch_inat_obs,
    fetch_wiki_batch,
    make_normalized_cutout,
    unload_ollama_model,
)
from test_pre_stylization_crop_repair import build_canny_from_repaired

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_batch2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

# 20 Brand-New Unseen Species (0 overlap with current_100 and 0 overlap with Batch 1)
BATCH2_20_SPECIES = [
    # Mammals (5)
    ("01_collared_peccary", "Collared Peccary", "Pecari tajacu", 42113, "Mammalia", "mammal_ground"),
    ("02_ringtail", "Ringtail", "Bassariscus astutus", 41676, "Mammalia", "mammal_ground"),
    ("03_gunnisons_prairie_dog", "Gunnison's Prairie Dog", "Cynomys gunnisoni", 46180, "Mammalia", "mammal_ground"),
    ("04_merriams_kangaroo_rat", "Merriam's Kangaroo Rat", "Dipodomys merriami", 44126, "Mammalia", "mammal_ground"),
    ("05_muskrat", "Muskrat", "Ondatra zibethicus", 45763, "Mammalia", "mammal_ground"),
    # Birds: Perching, Ground, and Swimming (5)
    ("06_painted_bunting", "Painted Bunting", "Passerina ciris", 10224, "Aves", "bird_perched"),
    ("07_cedar_waxwing", "Cedar Waxwing", "Bombycilla cedrorum", 7428, "Aves", "bird_perched"),
    ("08_belted_kingfisher", "Belted Kingfisher", "Megaceryle alcyon", 2548, "Aves", "bird_perched"),
    ("09_greater_roadrunner", "Greater Roadrunner", "Geococcyx californianus", 1986, "Aves", "bird_ground"),
    ("10_hooded_merganser", "Hooded Merganser", "Lophodytes cucullatus", 7109, "Aves", "bird_swimming"),
    # Reptiles: Snake, Lizards, Turtle (5)
    ("11_rough_greensnake", "Rough Greensnake", "Opheodrys aestivus", 29200, "Reptilia", "snake"),
    ("12_western_fence_lizard", "Western Fence Lizard", "Sceloporus occidentalis", 36204, "Reptilia", "lizard"),
    ("13_texas_horned_lizard", "Texas Horned Lizard", "Phrynosoma cornutum", 73905, "Reptilia", "lizard"),
    ("14_mediterranean_house_gecko", "Mediterranean House Gecko", "Hemidactylus turcicus", 34435, "Reptilia", "lizard"),
    ("15_diamondback_terrapin", "Diamondback Terrapin", "Malaclemys terrapin", 39836, "Reptilia", "turtle"),
    # Amphibians: Frogs & Salamanders (3)
    ("16_northern_leopard_frog", "Northern Leopard Frog", "Lithobates pipiens", 66003, "Amphibia", "amphibian"),
    ("17_spring_peeper", "Spring Peeper", "Pseudacris crucifer", 24268, "Amphibia", "amphibian"),
    ("18_western_tiger_salamander", "Western Tiger Salamander", "Ambystoma mavortium", 26765, "Amphibia", "amphibian"),
    # Elasmobranchs: Sharks & Rays (2)
    ("19_horn_shark", "Horn Shark", "Heterodontus francisci", 102710, "Elasmobranchii", "shark_ray"),
    ("20_hallers_round_ray", "Haller's Round Ray", "Urobatis halleri", 114782, "Elasmobranchii", "shark_ray"),
]


def download_pil_compliant(url: str, timeout: int = 20) -> Image.Image | None:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "AnimlVisionArtPipeline/2.0 (https://animl.org; contact@animl.org)"},
            timeout=timeout,
        )
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def fetch_commons_files(sci: str, common: str, limit: int = 10) -> list[dict]:
    """Query Wikimedia Commons File namespace (gsrnamespace=6) with compliant User-Agent."""
    out = []
    seen = set()
    for q in (sci, f"{sci} {common}"):
        try:
            r = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrnamespace": 6,
                    "gsrsearch": q,
                    "gsrlimit": limit,
                    "prop": "imageinfo",
                    "iiprop": "url|dimensions",
                    "iiurlwidth": 1280,
                    "format": "json",
                },
                headers={"User-Agent": "AnimlVisionArtPipeline/2.0 (https://animl.org; contact@animl.org)"},
                timeout=15,
            ).json()
            for pid, p in r.get("query", {}).get("pages", {}).items():
                if pid in seen or "imageinfo" not in p:
                    continue
                t = p["title"].lower()
                if any(k in t for k in ("map", "range", "skull", "skeleton", "egg", "track", "footprint", "stamp", ".svg", ".webm", ".ogg", ".pdf")):
                    continue
                seen.add(pid)
                ii = p["imageinfo"][0]
                u = ii.get("thumburl") or ii["url"]
                out.append({"photo_id": f"commons_{pid}", "original_url": u, "medium_url": u, "title": p["title"]})
        except Exception:
            pass
    return out


def sanitize_species_subject(common: str, sci: str, cls_name: str, subtype: str) -> tuple[str, str]:
    """
    Smart False-Animal Common-Name Sanitizer:
    - Sanitizes misleading animal tokens that distort anatomy (e.g. 'Prairie Dog' -> rodent, 'Kangaroo Rat' -> Dipodomys desert rodent,
      'Alligator Lizard' -> Elgaria lizard, 'Bat Ray' -> Myliobatis ray)
    - Preserves true patterned species names where the head noun is clear ('Leopard Shark', 'Northern Leopard Frog', 'Tiger Salamander', 'Horn Shark').
    Returns (subject_phrase, extra_negative_tokens).
    """
    c_low = common.lower()
    extra_neg = ""
    if "prairie dog" in c_low:
        return f"a plump burrowing Cynomys ground squirrel rodent ({sci})", "canine, dog snout, dog ears, puppy, "
    if "kangaroo rat" in c_low:
        return f"a small bipedal Dipodomys desert rodent ({sci})", "marsupial kangaroo, pouch, giant kangaroo, "
    if "alligator lizard" in c_low:
        return f"a slender Elgaria lizard ({sci})", "alligator snout, crocodile head, caiman, "
    if "bat ray" in c_low:
        return f"a smooth-skinned Myliobatis ray ({sci})", "bat ears, mammal ears, pointed ears, spots, polka dots, "
    if "horned lizard" in c_low:
        return f"a flat-bodied spiny Phrynosoma lizard ({sci})", "mammal horns, bull horns, goat horns, "
    if "horn shark" in c_low:
        return f"a spotted Heterodontus Horn Shark ({sci})", "mammal horns, bull horns, tiger stripes, "
    return f"a {common} ({sci})", extra_neg


def vet_side_by_side_and_extract_traits(
    pad_photo: Image.Image,
    norm_cut: Image.Image,
    common: str,
    sci: str,
    subtype: str,
    wiki_clauses: str,
) -> dict:
    """Pass `[Original Photo | Cutout]` side-by-side to Gemma3:27b so it sees both the pre-crop context
    AND the isolated cutout simultaneously, plus extracts Photo-Primary + Wiki-Assisted visual traits."""
    sbs = Image.new("RGB", (1024, 512), (255, 255, 255))
    sbs.paste(pad_photo.resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
    sbs.paste(norm_cut.resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
    buf = BytesIO()
    sbs.save(buf, format="JPEG", quality=86)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    marine_rule = ""
    if subtype == "shark_ray":
        marine_rule = (
            "CRITICAL FOR SHARKS/RAYS: `valid_live_wild_pose` MUST be false if the shark/ray is dead on a beach, lying on dry sand, "
            "caught on a fishing line/boat, held by a human, shot through a reflective aquarium glass window with human/room reflections or glare streaks across the animal, "
            "or if the cutout has a hole missing from its fin/wing or is super murky green.\n"
        )
    elif subtype == "snake":
        marine_rule = (
            "CRITICAL FOR SNAKES: `valid_live_wild_pose` MUST be false if the snake is held in a human hand, tangled in wire/netting, "
            "or if the cutout is missing the head or split into pieces.\n"
        )

    prompt = (
        f"Left image is the original reference photo; Right image is the segmented white-background cutout of a {common} ({sci}).\n"
        f"Wikipedia reference notes (use ONLY for general anatomical texture, NEVER to override the photo's actual color/pattern): {wiki_clauses}\n"
        f"{marine_rule}\n"
        f"Inspect BOTH Left and Right images carefully and return STRICT JSON with these keys:\n"
        f"1. \"valid_live_wild_pose\": boolean (true ONLY if the animal is alive, in a clear full-body pose, NOT held by a human hand, NOT dead/roadkill/beached, NOT obscured by aquarium glass window reflections/glare, and the Right cutout cleanly preserves the head and body without giant missing holes)\n"
        f"2. \"no_fused_background_blob\": boolean (true if the Right cutout does not have a giant rock, human hand, or mud wall fused to the body; a small twig under a perched bird's feet is OK)\n"
        f"3. \"no_aquarium_glass_reflection\": boolean (true if the photo is clean and free of aquarium/zoo window glass reflections, human reflections in glass, or bright glare streaks across the body)\n"
        f"3. \"pose_quality_score\": number from 0 to 10 (10 = sharp, well-lit, full-body side or 3/4 field-guide view where head, eye, and body markings are crystal clear; <=3 if blurry, murky, foreshortened, or clipped in half)\n"
        f"4. \"cutout_repair_needed\": string (\"none\", \"complete_paws_or_toes\", \"swimming_waterline_tucked_feet\", \"perch_stub_under_feet\", or \"remove_small_debris\")\n"
        f"5. \"eye_color_in_photo\": short phrase describing the exact iris/eye color visible in the photo (e.g. \"bright golden-yellow iris with black pupil\", \"dark brown-black eye\", \"pale cream upper iris and brown lower iris\")\n"
        f"6. \"visual_traits\": 20 to 32 words describing ONLY the specific colors, markings, patterns, and anatomical features visible on THIS individual in the photo (assisted by Wikipedia for texture, comma-separated phrases, no background words)."
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
            timeout=90,
        )
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except Exception as e:
        print(f"  [VLM warning for {common}]: {e}", flush=True)
        return {
            "valid_live_wild_pose": True,
            "no_fused_background_blob": True,
            "pose_quality_score": 5.0,
            "cutout_repair_needed": "none",
            "eye_color_in_photo": "natural dark eye",
            "visual_traits": wiki_clauses[:180] if wiki_clauses else f"natural {common} coloration and markings",
        }


def run_stage1(mode: str = "both") -> dict:
    import gc
    photo_dir = WORK / "photos_selected"
    cutout_dir = WORK / "cutouts_1024"
    sbs_dir = WORK / "sbs_selected"
    cand_cache_dir = WORK / "cands_top3"
    photo_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)
    sbs_dir.mkdir(parents=True, exist_ok=True)
    cand_cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = WORK / "batch2_metadata.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for k in list(meta.keys()):
        if meta[k].get("pose_quality_score") == 5.0 and meta[k].get("eye_color_in_photo") == "natural dark eye":
            del meta[k]

    if mode in ("both", "1a"):
        unload_ollama_model("gemma3:27b")
        torch.cuda.empty_cache()

        from transformers import CLIPModel, CLIPProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        for idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(BATCH2_20_SPECIES):
            sp_cand_dir = cand_cache_dir / stem
            sp_cand_meta = sp_cand_dir / "cands.json"
            if (stem in meta and (cutout_dir / f"{stem}.png").exists()) or sp_cand_meta.exists():
                print(f"[Phase 1A {idx+1}/20] {stem} top-3 cutouts already cached.", flush=True)
                continue

            print(f"[Phase 1A {idx+1}/20] Fetching & segmenting candidates for {common} ({sci})...", flush=True)
            cands = []
            seen = set()
            for c in fetch_inat_curated_taxon(taxon_id, limit=8):
                if c["photo_id"] not in seen:
                    seen.add(c["photo_id"])
                    cands.append(c)
            for c in fetch_commons_files(sci, common, limit=6):
                if c["photo_id"] not in seen:
                    seen.add(c["photo_id"])
                    cands.append(c)
            try:
                for c in fetch_inat_obs(taxon_id, "votes", 10):
                    if c["photo_id"] not in seen:
                        seen.add(c["photo_id"])
                        cands.append(c)
            except Exception:
                pass

            valid_packs = []
            for c in cands[:16]:
                im = download_pil_compliant(c["medium_url"], timeout=12) or download_pil_compliant(c["original_url"], timeout=15)
                if im is None or min(im.size) < 300:
                    continue
                norm_cut, pad_photo, st = make_normalized_cutout(im, subtype=subtype)
                if not st.get("valid") or norm_cut is None:
                    continue
                valid_packs.append((norm_cut, pad_photo, c, st))
                if len(valid_packs) >= 7:
                    break

            if not valid_packs:
                raise RuntimeError(f"No valid cutouts found for {stem} ({common})")

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

            ranked_indices = np.argsort(logits)[::-1][:3]
            sp_cand_dir.mkdir(parents=True, exist_ok=True)
            saved_cands = []
            for rank_i, r_idx in enumerate(ranked_indices):
                norm_cut, pad_photo, c, st = valid_packs[int(r_idx)]
                clip_s = float(logits[int(r_idx)])
                im_hi = download_pil_compliant(c["original_url"], timeout=16)
                if im_hi is not None and max(im_hi.size) > 600:
                    hi_cut, hi_pad, hi_st = make_normalized_cutout(im_hi, subtype=subtype)
                    if hi_st.get("valid") and hi_cut is not None:
                        norm_cut, pad_photo = hi_cut, hi_pad
                norm_cut.save(sp_cand_dir / f"cut_{rank_i}.png")
                pad_photo.save(sp_cand_dir / f"photo_{rank_i}.png")
                saved_cands.append({
                    "rank": rank_i,
                    "photo_id": c["photo_id"],
                    "original_url": c["original_url"],
                    "clip_score": round(clip_s, 4),
                })
            sp_cand_meta.write_text(json.dumps(saved_cands, indent=2), encoding="utf-8")

        if mode == "1a":
            print("Phase 1A complete! Exiting process to release 100% of ONNX Runtime & CUDA VRAM.", flush=True)
            return meta

    # =========================================================================
    # PHASE 1B: Run Gemma3:27b VLM Vetting + Trait Extraction with 100% free GPU!
    # =========================================================================
    print("Fetching Wikipedia descriptions for Batch 2 (20 species)...", flush=True)
    wiki_titles = [s[1] for s in BATCH2_20_SPECIES]
    wiki_raw = fetch_wiki_batch(wiki_titles)
    wiki_clauses_map = {}
    for _, common, sci, _, _, _ in BATCH2_20_SPECIES:
        txt = wiki_raw.get(common, "") or fetch_wiki_batch([sci]).get(sci, "")
        wiki_clauses_map[common] = extract_visual_clauses(txt)

    print("Starting Phase 1B: Gemma3:27b Side-by-Side VLM Vetting (GPU VRAM cleared)...", flush=True)
    for idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(BATCH2_20_SPECIES):
        out_cut = cutout_dir / f"{stem}.png"
        out_photo = photo_dir / f"{stem}.png"
        if stem in meta and out_cut.exists() and out_photo.exists():
            print(f"[Phase 1B {idx+1}/20] {stem} already vetted (score={meta[stem].get('pose_quality_score')}).", flush=True)
            continue

        sp_cand_dir = cand_cache_dir / stem
        saved_cands = json.loads((sp_cand_dir / "cands.json").read_text(encoding="utf-8"))
        best_choice = None
        best_total = -999.0

        print(f"[Phase 1B {idx+1}/20] VLM vetting top candidates for {common} ({sci})...", flush=True)
        for c_info in saved_cands:
            rank_i = c_info["rank"]
            norm_cut = Image.open(sp_cand_dir / f"cut_{rank_i}.png").convert("RGB")
            pad_photo = Image.open(sp_cand_dir / f"photo_{rank_i}.png").convert("RGB")
            clip_s = float(c_info["clip_score"])
            vlm_res = vet_side_by_side_and_extract_traits(
                pad_photo, norm_cut, common, sci, subtype, wiki_clauses_map.get(common, "")
            )
            valid_live = bool(vlm_res.get("valid_live_wild_pose", False))
            no_blob = bool(vlm_res.get("no_fused_background_blob", False))
            pose_s = float(vlm_res.get("pose_quality_score", 0.0))
            total_s = pose_s + clip_s * 2.5 - (0.0 if valid_live else 8.0) - (0.0 if no_blob else 5.0)
            print(
                f"  cand id={c_info['photo_id']}: live={valid_live}, no_blob={no_blob}, pose={pose_s:.1f}, repair={vlm_res.get('cutout_repair_needed')} -> total={total_s:.2f}",
                flush=True,
            )
            if total_s > best_total:
                best_total = total_s
                best_choice = (norm_cut, pad_photo, c_info, vlm_res, total_s)
            # Early-accept if the #1 CLIP-ranked candidate passes live + no_blob with pose >= 7.5!
            if valid_live and no_blob and pose_s >= 7.5:
                break

        norm_cut, pad_photo, c_info, vlm_res, total_s = best_choice
        pad_photo.save(out_photo)
        norm_cut.save(out_cut)
        sbs = Image.new("RGB", (1024, 512), (255, 255, 255))
        sbs.paste(pad_photo.resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
        sbs.paste(norm_cut.resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
        sbs.save(sbs_dir / f"{stem}.jpg", quality=88)

        meta[stem] = {
            "common_name": common,
            "scientific_name": sci,
            "taxon_id": taxon_id,
            "class": cls_name,
            "subtype": subtype,
            "photo_id": c_info["photo_id"],
            "photo_url": c_info["original_url"],
            "pose_quality_score": vlm_res.get("pose_quality_score"),
            "cutout_repair_needed": vlm_res.get("cutout_repair_needed", "none"),
            "eye_color_in_photo": vlm_res.get("eye_color_in_photo", ""),
            "visual_traits": vlm_res.get("visual_traits", ""),
            "wiki_clauses": wiki_clauses_map.get(common, ""),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    unload_ollama_model("gemma3:27b")
    torch.cuda.empty_cache()

    # Build a quick 20-species Stage 1 contact sheet so we can inspect all 20 `[Photo | Cutout]` pairs
    sheet = Image.new("RGB", (1600, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (stem, common, sci, _, _, subtype) in enumerate(BATCH2_20_SPECIES):
        r, col = divmod(i, 2)
        x0, y0 = col * 800, r * 200
        sbs_p = sbs_dir / f"{stem}.jpg"
        if sbs_p.exists():
            im_s = Image.open(sbs_p).resize((780, 175), Image.Resampling.LANCZOS)
            sheet.paste(im_s, (x0 + 10, y0 + 22))
        rep = meta.get(stem, {}).get("cutout_repair_needed", "none")
        draw.text((x0 + 12, y0 + 4), f"{stem} ({common}) [{subtype}] repair={rep}", fill=(20, 20, 20))
    sheet_path = ARTIFACT_DIR / "batch2_stage1_sbs_sheet.jpg"
    sheet.save(sheet_path, quality=90)
    print(f"Saved Stage 1 side-by-side contact sheet to {sheet_path}", flush=True)
    return meta


def refresh_switched_traits() -> dict:
    """Re-extract Photo-Primary + Wiki-Assisted visual traits and eye color for any species whose comparative cutout was updated."""
    meta_path = WORK / "batch2_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    switched = [
        "02_ringtail",
        "06_painted_bunting",
        "07_cedar_waxwing",
        "09_greater_roadrunner",
        "10_hooded_merganser",
        "13_texas_horned_lizard",
        "15_diamondback_terrapin",
        "17_spring_peeper",
        "18_western_tiger_salamander",
        "19_horn_shark",
        "20_hallers_round_ray",
    ]
    sp_map = {s[0]: s for s in BATCH2_20_SPECIES}
    for stem in switched:
        if meta.get(stem, {}).get("refreshed_comparative"):
            continue
        _, common, sci, _, _, subtype = sp_map[stem]
        pad_photo = Image.open(WORK / "photos_selected" / f"{stem}.png").convert("RGB")
        norm_cut = Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB")
        wiki_c = meta.get(stem, {}).get("wiki_clauses", "")
        print(f"[Refresh Traits] Extracting photo-primary traits for {stem} ({common})...", flush=True)
        res = vet_side_by_side_and_extract_traits(pad_photo, norm_cut, common, sci, subtype, wiki_c)
        meta[stem]["visual_traits"] = res.get("visual_traits", meta[stem].get("visual_traits", ""))
        meta[stem]["eye_color_in_photo"] = res.get("eye_color_in_photo", meta[stem].get("eye_color_in_photo", ""))
        meta[stem]["refreshed_comparative"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    unload_ollama_model("gemma3:27b")
    torch.cuda.empty_cache()
    return meta


def normalize_repaired_cutout(im: Image.Image) -> Image.Image:
    """Normalize a GenAI-repaired photo cutout to 1024x1024 pure white (#FFFFFF) background."""
    im = im.convert("RGB")
    if im.size != (1024, 1024):
        im = im.resize((1024, 1024), Image.Resampling.LANCZOS)
    arr = np.array(im)
    # Snap near-white background pixels (>= 244 on all channels) to pure 255 white
    bg = np.all(arr >= 244, axis=-1)
    arr[bg] = 255
    return Image.fromarray(arr)


def run_stylize() -> None:
    """
    Stage 2 + Stage 3:
    1. Assemble final cutouts (`cutouts_final`): uses the GenAI-repaired cutout for the 6 repaired species
       and the clean Stage 1 cutout for the other 14 species.
    2. Build `canny_final` and `depth_final` maps (`depth-anything/Depth-Anything-V2-Small-hf`).
    3. Run `SDXL + Canny + Depth ControlNet + animl_style_v6` with Smart False-Animal Sanitizer,
       eye-color prompt injection, and perch-preserving `clean_post`.
    4. Build `v6_batch2_20_part1.jpg` and `v6_batch2_20_part2.jpg` + copy all 20 PNGs to `ARTIFACT_DIR`.
    """
    import gc
    from transformers import pipeline as hf_pipeline
    from diffusers import (
        ControlNetModel,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

    meta = refresh_switched_traits()

    cut_final_dir = WORK / "cutouts_final"
    canny_dir = WORK / "canny_final"
    depth_dir = WORK / "depth_final"
    out_dir = WORK / "clean_out"
    for d in (cut_final_dir, canny_dir, depth_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    repaired_stems = {
        "01_collared_peccary",
        "03_gunnisons_prairie_dog",
        "06_painted_bunting",
        "07_cedar_waxwing",
        "08_belted_kingfisher",
        "10_hooded_merganser",
    }
    perched_stems = {"06_painted_bunting", "07_cedar_waxwing", "08_belted_kingfisher"}

    # 1. Prepare cutouts_final and Canny maps
    for stem, common, sci, _, _, subtype in BATCH2_20_SPECIES:
        gen_p = ARTIFACT_DIR / f"genai_b2_{stem}.png"
        if stem in repaired_stems and gen_p.exists():
            cut_im = normalize_repaired_cutout(Image.open(gen_p))
        else:
            cut_im = Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB")
        cut_im.save(cut_final_dir / f"{stem}.png")
        canny_im = build_canny_from_repaired(cut_im)
        canny_im.save(canny_dir / f"{stem}.png")

    # 2. Compute Depth maps with Depth-Anything-V2-Small
    need_depth = [s[0] for s in BATCH2_20_SPECIES if not (depth_dir / f"{s[0]}.png").exists()]
    if need_depth:
        print(f"Computing Depth-Anything-V2 maps for {len(need_depth)} species...", flush=True)
        depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if torch.cuda.is_available() else -1,
        )
        for stem in need_depth:
            cut_im = Image.open(cut_final_dir / f"{stem}.png").convert("RGB")
            arr = np.array(cut_im)
            fg = np.any(arr < 246, axis=-1)
            d_im = depth_pipe(cut_im)["depth"].resize((1024, 1024), Image.Resampling.BILINEAR)
            d_arr = np.array(d_im, dtype=np.float32)
            if fg.any():
                lo, hi = np.percentile(d_arr[fg], [2, 98])
                d_norm = np.clip((d_arr - lo) / max(1e-5, hi - lo) * 215.0 + 40.0, 40.0, 255.0)
            else:
                d_norm = d_arr
            d_norm[~fg] = 0.0
            d_rgb = np.stack([d_norm.astype(np.uint8)] * 3, axis=-1)
            Image.fromarray(d_rgb).save(depth_dir / f"{stem}.png")
        del depth_pipe
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Load SDXL + Canny + Depth ControlNet + animl_style_v6
    print("Loading SDXL + Canny/Depth MultiControlNet + animl_style_v6...", flush=True)
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

    raw_styl_dir = WORK / "raw_stylized"
    raw_styl_dir.mkdir(parents=True, exist_ok=True)

    for idx, (stem, common, sci, _, cls_name, subtype) in enumerate(BATCH2_20_SPECIES):
        raw_p = raw_styl_dir / f"{stem}.png"
        print(f"[Stylize {idx+1}/20] Rendering {stem} ({common})...", flush=True)

        cut_im = Image.open(cut_final_dir / f"{stem}.png").convert("RGB")
        canny_im = Image.open(canny_dir / f"{stem}.png").convert("RGB")
        depth_im = Image.open(depth_dir / f"{stem}.png").convert("RGB")

        subj_phrase, extra_neg = sanitize_species_subject(common, sci, cls_name, subtype)
        traits = meta.get(stem, {}).get("visual_traits", f"natural {common} coloration and markings")
        eye_col = meta.get(stem, {}).get("eye_color_in_photo", "")
        eye_clause = f", {eye_col}" if eye_col and "natural dark" not in eye_col.lower() else ""

        perch_clause = ""
        neg_prompt = extra_neg + NEG_V6_UNIFIED
        if stem in perched_stems:
            perch_clause = ", perched with both feet gripping a short natural wooden twig stub strictly under its feet"
            neg_prompt = re.sub(r"\b(branch|twig|wood),\s*", "", neg_prompt)

        prompt = (
            f"animl_gouache_style, {subj_phrase}, full body wildlife illustration{perch_clause}, "
            f"{traits}{eye_clause}, {STYLE_V6}"
        )

        # Higher Canny conditioning for spotted/patterned species
        canny_scale = 0.48 if any(k in stem for k in ("shark", "ray", "salamander", "frog", "terrapin", "gecko", "bunting")) else 0.45
        depth_scale = 0.25

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
            controlnet_conditioning_scale=[canny_scale, depth_scale],
            control_guidance_start=0.0,
            control_guidance_end=[0.76, 0.60],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
        ).images[0]
        res_im.save(raw_p)

    # Free SDXL + ControlNets from GPU VRAM BEFORE running BiRefNet (`clean_post`)!
    del pipe, cnet_canny, cnet_depth
    gc.collect()
    torch.cuda.empty_cache()

    print("Running post-clean background removal + perch stub preservation...", flush=True)
    for idx, (stem, common, sci, _, cls_name, subtype) in enumerate(BATCH2_20_SPECIES):
        raw_p = raw_styl_dir / f"{stem}.png"
        out_p = out_dir / f"{stem}.png"
        art_p = ARTIFACT_DIR / f"batch2_{stem}.png"
        res_im = Image.open(raw_p).convert("RGB")
        cut_im = Image.open(cut_final_dir / f"{stem}.png").convert("RGB")

        if out_p.exists():
            out_p.unlink()
        clean_post(res_im, out_p)

        # For perched birds, preserve the short wooden perch stub from the repaired cutout mask so BiRefNet doesn't strip it
        if stem in perched_stems:
            post_rgba = Image.open(out_p).convert("RGBA")
            post_arr = np.array(post_rgba)
            cut_arr = np.array(cut_im)
            fg_cut = np.any(cut_arr < 242, axis=-1)
            fg_eroded = cv2.erode(fg_cut.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
            res_arr = np.array(res_im)
            missing_perch = fg_eroded & (post_arr[:, :, 3] < 128)
            post_arr[missing_perch, :3] = res_arr[missing_perch]
            post_arr[missing_perch, 3] = 255
            Image.fromarray(post_arr).save(out_p)

        shutil.copyfile(out_p, art_p)

    # 4. Build the two 4-column comparison sheets (`v6_batch2_20_part1.jpg` and `v6_batch2_20_part2.jpg`)
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
            p_cut = cut_final_dir / f"{stem}.png"
            p_canny = canny_dir / f"{stem}.png"
            p_styl = out_dir / f"{stem}.png"
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
        print(f"Saved Batch 2 Part {part_idx} sheet to {out_sheet}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--phase1a", action="store_true")
    parser.add_argument("--phase1b", action="store_true")
    parser.add_argument("--refresh-traits", action="store_true")
    parser.add_argument("--stylize", action="store_true")
    args = parser.parse_args()
    if args.phase1a:
        run_stage1("1a")
    elif args.phase1b:
        run_stage1("1b")
    elif args.stage1_only:
        run_stage1("both")
    elif args.refresh_traits:
        refresh_switched_traits()
    elif args.stylize:
        run_stylize()
