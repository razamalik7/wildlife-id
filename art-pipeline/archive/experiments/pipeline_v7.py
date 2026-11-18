"""Shared v7 wildlife illustration pipeline helpers (general prompt, VLM trait, and perch-clean fixes).

Fixes applied generally across all species (no per-species hardcoding hacks):
1. False-animal common-name sanitization without colliding with v7 trained squirrel tokens
   (e.g., Prairie Dog maps to 'unstriped ... Cynomys rodent' rather than 'ground squirrel').
2. Gemma3:27b side-by-side trait extraction prioritizing namesake/diagnostic field marks
   and explicitly stating 'unstriped' when a mammal coat has no stripes.
3. Negative-prompt guards: preserve legitimate pale collars/bands/stripes, block chipmunk
   striping on unstriped mammals, and block wing-red bleed onto bird crowns/eyebrows.
4. Perch-preserving `clean_post_v7`: restores BiRefNet-stripped perches from raw SDXL / cel
   cutout BEFORE bounding-box cropping and RGB flattening, plus 1:1 non-resampling centering.
"""

import base64
import json
import re
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import binary_propagation

from normalize import subject_bbox
from test_fresh_20_e2e import (
    NEG_V6_UNIFIED,
    OLLAMA_URL,
    STYLE_V6,
    unload_ollama_model,
)
from test_pre_stylization_crop_repair import build_canny_from_repaired
from u2net_matte import remove
from wiki_descriptions import extract_visual_clauses, fetch_wiki_batch

_RAY_FIN_EYE_NEG = (
    "translucent wing, transparent fin, fin rays, bony fish fin rays, "
    "operculum gill cover, giant fish eye, bulging fish eye, "
)

# (keyword_in_common_lower, required_class_or_None, phrase_fn(sci), extra_neg_prefix)
_SUBJECT_SANITIZER_RULES: list[tuple[str, str | None, object, str]] = [
    (
        "prairie dog",
        None,
        lambda sci: f"a plump unstriped buff-tan Cynomys prairie marmot ({sci})",
        "canine, dog snout, dog ears, puppy, chipmunk stripes, lateral stripes, dorsal stripes, striped back, black and white stripes, pink nose, pink snout, pink feet, hairless rat tail, rat, mouse, ",
    ),
    (
        "sea lion",
        None,
        lambda sci: f"a large golden-brown {sci.split()[0]} eared seal pinniped ({sci})",
        "lion mane, cat face, feline paws, terrestrial mammal legs, ",
    ),
    (
        "kangaroo rat",
        None,
        lambda sci: f"a small bipedal Dipodomys desert rodent ({sci})",
        "marsupial kangaroo, pouch, giant kangaroo, ",
    ),
    (
        "alligator lizard",
        None,
        lambda sci: f"a slender Elgaria lizard ({sci})",
        "alligator snout, crocodile head, caiman, ",
    ),
    (
        "bat ray",
        None,
        lambda sci: f"a smooth-skinned Myliobatis ray ({sci})",
        "bat ears, mammal ears, pointed ears, spots, polka dots, " + _RAY_FIN_EYE_NEG + "lateral shark gill slits, forked shark caudal fin, ",
    ),
    (
        "cownose ray",
        None,
        lambda sci: f"a smooth-winged Rhinoptera eagle ray ({sci})",
        "cow, bovine, mammal snout, horns, " + _RAY_FIN_EYE_NEG + "lateral shark gill slits, forked shark caudal fin, ",
    ),
    (
        "eagle ray",
        None,
        lambda sci: (
            f"a countershaded Aetobatus pelagic ray ({sci}) with white spots strictly on its dark navy dorsal back, "
            f"a solid unspotted opaque cream-white ventral wing underside and belly, and a long thin whip-like tail"
        ),
        (
            "bird feathers, eagle beak, talons, avian wings, black spots, dark spots on belly, "
            "spots on white underside, spotted underbelly, leopard spots, lateral pectoral fish fin, "
            + _RAY_FIN_EYE_NEG
            + "lateral shark gill slits, shark dorsal fin, forked shark caudal fin, "
        ),
    ),
    (
        "horned lizard",
        None,
        lambda sci: f"a flat-bodied spiny Phrynosoma lizard ({sci})",
        "mammal horns, bull horns, goat horns, ",
    ),
    (
        "horn shark",
        None,
        lambda sci: f"a brown spotted Heterodontus benthic shark ({sci})",
        "mammal horns, bull horns, tiger stripes, ",
    ),
    (
        "smooth-hound",
        "Elasmobranchii",
        lambda sci: f"a sleek Mustelus benthic shark ({sci})",
        "canine, dog, mammal fur, legs, ",
    ),
    (
        "hound",
        "Elasmobranchii",
        lambda sci: f"a sleek Mustelus benthic shark ({sci})",
        "canine, dog, mammal fur, legs, ",
    ),
    (
        "guitarfish",
        None,
        lambda sci: f"a flat-bodied benthic ray-like elasmobranch ({sci})",
        "guitar, musical instrument, strings, " + _RAY_FIN_EYE_NEG,
    ),
]


def is_batoidea_ray_v7(common: str, cls_name: str) -> bool:
    """True for cartilaginous rays, skates, mantas, and guitarfish (0 in dataset_v7; avoid reef-fish anatomy priors)."""
    return cls_name == "Elasmobranchii" and any(
        k in common.lower() for k in ("ray", "skate", "manta", "guitarfish", "sawfish")
    )


def sanitize_species_subject_v7(common: str, sci: str, cls_name: str, subtype: str) -> tuple[str, str]:
    """Rewrite misleading false-animal common names into clear taxonomic descriptors."""
    c_low = common.lower()
    for kw, req_cls, phrase_fn, extra_neg in _SUBJECT_SANITIZER_RULES:
        if req_cls is not None and cls_name != req_cls:
            continue
        if kw in c_low:
            return phrase_fn(sci), extra_neg
    if cls_name == "Elasmobranchii" and any(k in c_low for k in ("stingray", "manta", "devil ray", "ray", "skate")):
        return f"a {common} ({sci})", _RAY_FIN_EYE_NEG + "lateral shark gill slits, forked shark caudal fin, mammal fur, legs, "
    return f"a {common} ({sci})", ""


_MARINE_BAD_TITLE_WORDS = (
    "aquarium", "zoo", "jaws", "tooth", "teeth", "skeleton", "skull", "caught",
    "fisherman", "market", "dead", "beach", "map", "range", "stamp", ".svg", ".webm", ".ogg", ".pdf",
)


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


def fetch_marine_candidates_v7(taxon_id: int, sci: str, common: str) -> list[dict]:
    """Fetch & interleave global underwater scuba observations and Commons media for marine taxa."""
    headers = {"User-Agent": "AnimlVisionArtPipeline/2.0 (https://animl.org; contact@animl.org)"}
    src_taxa, src_obs, src_commons = [], [], []

    try:
        r = requests.get(f"https://api.inaturalist.org/v1/taxa/{taxon_id}", headers=headers, timeout=20).json()
        for res in r.get("results", [])[:1]:
            for tp in res.get("taxon_photos", [])[:10]:
                p = tp.get("photo", {})
                lic = (p.get("license_code") or "").lower()
                u = p.get("url") or p.get("medium_url") or ""
                if u and (not lic or lic in ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "pd")):
                    src_taxa.append(
                        {
                            "photo_id": f"inat_t_{p['id']}",
                            "medium_url": u.replace("square", "medium"),
                            "original_url": u.replace("square", "original"),
                        }
                    )
    except Exception:
        pass

    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "captive": "false",
                "order_by": "votes",
                "per_page": 18,
            },
            headers=headers,
            timeout=25,
        ).json()
        for obs in r.get("results", []):
            for p in obs.get("photos", [])[:1]:
                u = p.get("url") or ""
                if u:
                    src_obs.append(
                        {
                            "photo_id": f"inat_{p['id']}",
                            "medium_url": u.replace("square", "medium"),
                            "original_url": u.replace("square", "original"),
                        }
                    )
    except Exception:
        pass

    seen_comm: set[str] = set()
    for q in (f'"{sci}" underwater', f'"{common}" underwater', sci):
        try:
            r = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrnamespace": 6,
                    "gsrsearch": q,
                    "gsrlimit": 10,
                    "prop": "imageinfo",
                    "iiprop": "url|dimensions",
                    "iiurlwidth": 1280,
                    "format": "json",
                },
                headers=headers,
                timeout=15,
            ).json()
            for pid, p in r.get("query", {}).get("pages", {}).items():
                if pid in seen_comm or "imageinfo" not in p:
                    continue
                t_low = p["title"].lower()
                if any(bw in t_low for bw in _MARINE_BAD_TITLE_WORDS):
                    continue
                seen_comm.add(pid)
                ii = p["imageinfo"][0]
                u = ii.get("thumburl") or ii.get("url", "")
                if u:
                    src_commons.append({"photo_id": f"commons_{pid}", "medium_url": u, "original_url": u})
        except Exception:
            pass

    out: list[dict] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    max_len = max(len(src_obs), len(src_taxa), len(src_commons), 0)
    for i in range(max_len):
        for bucket in (src_obs, src_taxa, src_commons):
            if i < len(bucket):
                c = bucket[i]
                pid, u = str(c["photo_id"]), c["original_url"]
                if pid not in seen_ids and u not in seen_urls:
                    seen_ids.add(pid)
                    seen_urls.add(u)
                    out.append(c)
                    if len(out) >= 24:
                        return out
    return out


def fetch_wiki_clauses_map(species_list: list[tuple[str, str]]) -> dict[str, str]:
    """Fetch Wikipedia visual clauses for `[(common, sci), ...]`, falling back to scientific name on stubs/disambig."""
    commons = [c for c, _ in species_list]
    raw_common = fetch_wiki_batch(commons)
    out: dict[str, str] = {}
    need_sci: list[tuple[str, str]] = []
    for common, sci in species_list:
        txt = raw_common.get(common, "")
        clauses = extract_visual_clauses(txt) if len(txt) >= 500 and "may refer to:" not in txt[:300].lower() else ""
        if clauses:
            out[common] = clauses
        else:
            need_sci.append((common, sci))

    if need_sci:
        raw_sci = fetch_wiki_batch([s for _, s in need_sci])
        for common, sci in need_sci:
            txt = raw_sci.get(sci, "")
            out[common] = extract_visual_clauses(txt)
    return out


def vet_side_by_side_and_extract_traits_v7(
    pad_photo: Image.Image,
    norm_cut: Image.Image,
    common: str,
    sci: str,
    subtype: str,
    wiki_clauses: str,
) -> dict:
    """Pass `[Original Photo | Cutout]` to Gemma3:27b, prioritizing namesake marks & explicit unstriped coat."""
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


def build_prompt_and_neg_v7(
    common: str,
    sci: str,
    cls_name: str,
    subtype: str,
    traits: str,
    eye_color: str,
    is_perched: bool,
) -> tuple[str, str, float]:
    """Construct positive prompt, calibrated negative prompt, and adaptive img2img strength."""
    subj_phrase, extra_neg = sanitize_species_subject_v7(common, sci, cls_name, subtype)
    neg_base = NEG_V6_UNIFIED
    traits_low = traits.lower()

    # White/Cream Band Guard: keep pale fur collars, breast bands, stripes & marbling intact
    if any(tok in traits_low for tok in ("collar", "band", "white", "cream", "stripe", "stripes", "marbled", "spotted")):
        neg_base = neg_base.replace("stark white fur streaks, ", "specular glare, ")

    extra_parts: list[str] = []
    if extra_neg:
        extra_parts.append(extra_neg.strip(", "))

    # Countershaded Elasmobranch Guard: keep ventral wing/belly solid unspotted opaque white
    is_countershaded_elasmo_ventral = cls_name == "Elasmobranchii" and any(
        tok in traits_low for tok in ("white underbelly", "white ventral", "white belly", "white underside", "pale underside")
    )
    if is_countershaded_elasmo_ventral:
        traits = re.sub(
            r"\bwhite pectoral fins\b",
            "solid unspotted opaque white ventral wing underside",
            traits,
            flags=re.IGNORECASE,
        )
        traits_low = traits.lower()
        extra_parts.append(
            "spots on white belly, dark spots on underside, spotted underbelly, "
            "translucent wing, transparent fin, fin rays, bony fish operculum, gill cover"
        )
        if not any(tok in traits_low for tok in ("black spots", "dark spots", "black saddles", "dark blotches")):
            extra_parts.append("black spots, leopard spots")

    # Ray/Skate Eye Dedup Guard: suppress duplicate eye phrases that trigger bony reef-fish giant eyes
    suppress_dark_eye_clause = False
    if cls_name == "Elasmobranchii" and any(k in common.lower() for k in ("ray", "skate", "manta", "guitarfish")):
        traits = re.sub(
            r"\bdark(?:\s+brown)?\s+eyes?\b",
            "small recessed dark lateral eye",
            traits,
            flags=re.IGNORECASE,
        )
        traits_low = traits.lower()
        if eye_color and "dark" in eye_color.lower() and "bright" not in eye_color.lower():
            suppress_dark_eye_clause = True

    # Unstriped Mammal Guard
    if "unstriped" in traits_low or "unstriped" in subj_phrase.lower():
        extra_parts.append("chipmunk stripes, lateral stripes, striped back")

    # Wing/Body Red Bleed Guard for Birds: only apply when red is NOT on the bird's head/face
    if cls_name == "Aves" and re.search(r"\b(?:red|reddish|scarlet|crimson)\b", traits_low):
        head_tokens = ("head", "crown", "crest", "brow", "eyebrow", "forehead", "face", "mask", "throat", "nape")
        red_on_head = any(
            re.search(r"\b(?:red|reddish|scarlet|crimson)\b", clause)
            and any(ht in clause for ht in head_tokens)
            for clause in traits_low.split(",")
        )
        if not red_on_head:
            eye_has_red = bool(re.search(r"\b(?:red|reddish|scarlet|crimson|ruby|orange-red)\b", eye_color.lower()))
            if eye_has_red:
                extra_parts.append("red crown, red eyebrow, red forehead, red head patch")
            else:
                extra_parts.append("red crown, red eyebrow, red forehead, red head patch, red eye, red iris")

    extra_tokens: list[str] = []
    seen_tok: set[str] = set()
    for chunk in extra_parts:
        for tok in chunk.split(","):
            t = tok.strip()
            if t and t.lower() not in seen_tok:
                seen_tok.add(t.lower())
                extra_tokens.append(t)

    perch_clause = ""
    if is_perched:
        perch_clause = ", perched with both feet gripping a short natural wooden twig stub strictly under its feet"
        neg_base = re.sub(r"\b(branch|twig|wood),\s*", "", neg_base)

    neg_prompt = (", ".join(extra_tokens) + ", " + neg_base) if extra_tokens else neg_base

    eye_clause = (
        f", {eye_color}"
        if (not suppress_dark_eye_clause and eye_color and "natural dark" not in eye_color.lower())
        else ""
    )
    prompt = (
        f"animl_gouache_style, {subj_phrase}, full body wildlife illustration{perch_clause}, "
        f"single animal, {traits}{eye_clause}, {STYLE_V6}"
    )
    if is_countershaded_elasmo_ventral:
        strength = 0.44
    elif "collar" in traits_low or "shoulder stripe" in traits_low:
        strength = 0.46
    elif "breast band" in traits_low:
        strength = 0.50
    else:
        strength = 0.54
    return prompt, neg_prompt, strength


def _fg_blur_v7(x: np.ndarray, fg_f: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Gaussian blur strictly inside foreground mask `fg_f`."""
    return cv2.GaussianBlur(x * fg_f, (0, 0), sigma) / np.maximum(0.05, cv2.GaussianBlur(fg_f, (0, 0), sigma))


def prepare_conditioned_cutout_and_canny_v7(
    cut_im: Image.Image,
    cls_name: str,
    traits: str,
    base_canny: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Lift pale mammal collars and countershaded elasmobranch ventrals before SDXL img2img."""
    traits_low = traits.lower()
    is_mammal_collar = cls_name == "Mammalia" and any(
        tok in traits_low for tok in ("collar", "shoulder stripe", "shoulder band", "neck band", "white band", "cream band")
    )
    is_countershaded_elasmo = (
        cls_name == "Elasmobranchii"
        and any(tok in traits_low for tok in ("white underbelly", "white ventral", "white belly", "white underside", "pale underside", "cream underside"))
        and not any(tok in traits_low for tok in ("black spots", "dark spots", "black saddles", "dark blotches"))
    )
    if not (is_mammal_collar or is_countershaded_elasmo):
        return cut_im, (base_canny if base_canny is not None else build_canny_from_repaired(cut_im))

    arr = np.array(cut_im.convert("RGB"))
    fg = np.any(arr < 242, axis=-1)
    fg_f = fg.astype(np.float32)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    if is_mammal_collar:
        fg_inner = cv2.erode(fg.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        fg_med = float(np.median(l_chan[fg_inner])) if fg_inner.any() else 110.0
        l_mean_loc = _fg_blur_v7(l_chan, fg_f, 12.0)
        l_mean_bg = _fg_blur_v7(l_chan, fg_f, 55.0)
        prom_mean = l_mean_loc - np.maximum(l_mean_bg, fg_med)

        k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        l_close = cv2.morphologyEx(np.where(fg, l_chan, fg_med), cv2.MORPH_CLOSE, k11)
        l_close_sm = _fg_blur_v7(l_close, fg_f, 4.0)

        w = np.zeros_like(l_chan)
        mask = fg_inner & (prom_mean > 8.0) & (l_mean_loc > fg_med + 14.0)
        w[mask] = np.clip((prom_mean[mask] - 6.0) / 14.0, 0.0, 0.92)
        w = cv2.GaussianBlur(w, (0, 0), 4.0) * fg_inner.astype(np.float32)

        lab_new = lab.copy()
        lab_new[:, :, 0] = np.clip(l_chan * (1.0 - w) + np.clip(l_close_sm * 1.15 + 16.0, 0, 226.0) * w, 0, 255)
        lab_new[:, :, 1] = lab[:, :, 1] * (1.0 - 0.55 * w) + 128.0 * (0.55 * w)
        lab_new[:, :, 2] = lab[:, :, 2] * (1.0 - 0.40 * w) + 136.0 * (0.40 * w)
        cond_rgb = cv2.cvtColor(lab_new.astype(np.uint8), cv2.COLOR_LAB2RGB)
        cond_rgb[~fg] = 255

        blurred_rgb = cv2.GaussianBlur(cond_rgb, (0, 0), 5.0).astype(np.float32)
        cond_for_canny = (
            cond_rgb.astype(np.float32) * (1.0 - 0.75 * w[:, :, None]) + blurred_rgb * (0.75 * w[:, :, None])
        ).astype(np.uint8)
        cond_for_canny[~fg] = 255
        canny_im = build_canny_from_repaired(Image.fromarray(cond_for_canny))
        return Image.fromarray(cond_rgb), canny_im

    # Countershaded dark-backed Elasmobranch: lift smooth shaded ventral wing/belly to opaque cream-white
    dorsal_p25 = float(np.percentile(l_chan[fg], 25)) if fg.any() else 100.0
    if dorsal_p25 >= 75.0:
        return cut_im, (base_canny if base_canny is not None else build_canny_from_repaired(cut_im))

    l_fg0 = np.where(fg, l_chan, 0.0)
    l_open19 = cv2.morphologyEx(l_fg0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    l_open9 = cv2.morphologyEx(l_fg0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    seed_v = fg & (l_open19 > max(80.0, dorsal_p25 + 42.0)) & (l_chan > max(85.0, dorsal_p25 + 47.0))
    supp_v = fg & (l_open9 > max(76.0, dorsal_p25 + 36.0)) & (l_chan > max(85.0, dorsal_p25 + 45.0))
    if not seed_v.any():
        return cut_im, (base_canny if base_canny is not None else build_canny_from_repaired(cut_im))

    ventral_mask = binary_propagation(seed_v, structure=np.ones((3, 3), dtype=bool), mask=supp_v)
    w = cv2.GaussianBlur(ventral_mask.astype(np.float32), (0, 0), 2.0) * fg_f
    rim_guard = np.clip((l_chan - (dorsal_p25 + 30.0)) / 22.0, 0.0, 1.0)
    w = w * rim_guard * 0.95

    l_smooth = _fg_blur_v7(l_chan, fg_f, 10.0)
    v_lo, v_hi = np.percentile(l_smooth[ventral_mask], [10, 90])
    t = np.clip((l_smooth - v_lo) / max(12.0, v_hi - v_lo), 0.0, 1.0)
    l_target = 202.0 + 16.0 * t

    lab_new = lab.copy()
    lab_new[:, :, 0] = np.clip(l_chan * (1.0 - w) + l_target * w, 0, 255)
    lab_new[:, :, 1] = lab[:, :, 1] * (1.0 - 0.88 * w) + 128.0 * (0.88 * w)
    lab_new[:, :, 2] = lab[:, :, 2] * (1.0 - 0.88 * w) + 130.0 * (0.88 * w)
    cond_rgb = cv2.cvtColor(lab_new.astype(np.uint8), cv2.COLOR_LAB2RGB)
    cond_rgb[~fg] = 255

    canny_src = base_canny if base_canny is not None else build_canny_from_repaired(cut_im)
    canny_arr = np.array(canny_src.convert("RGB")).copy()
    fg_inner7 = cv2.erode(fg.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    v_inner5 = cv2.erode(ventral_mask.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    canny_arr[fg_inner7 & v_inner5] = 0
    return Image.fromarray(cond_rgb), Image.fromarray(canny_arr)


def clean_post_v7(
    raw_im: Image.Image,
    cut_im: Image.Image,
    dst_path: Path,
    is_perched: bool = False,
    preserve_thin_appendages: bool = False,
) -> Image.Image:
    """Isolate stylized subject via BiRefNet BEFORE crop/flatten, preserving perches and thin whip tails."""
    cut_rgba = remove(raw_im.convert("RGBA"))
    arr = np.array(cut_rgba)
    alpha = arr[:, :, 3]
    seed_mask = alpha >= 165

    if seed_mask.any():
        low_sup = cv2.morphologyEx((alpha >= 45).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        prop_mask = binary_propagation(seed_mask, structure=np.ones((3, 3), dtype=bool), mask=low_sup)
        arr[~prop_mask, 3] = 0
        boost = prop_mask & (alpha < 220)
        arr[boost, 3] = np.clip(alpha[boost].astype(np.int32) * 3, 185, 255).astype(np.uint8)
    else:
        arr[alpha < 165, 3] = 0

    if is_perched or preserve_thin_appendages:
        cut_arr = np.array(cut_im.convert("RGB"))
        raw_arr = np.array(raw_im.convert("RGB"))
        dark_mask = (cut_arr.min(axis=-1) < 210).astype(np.uint8) * 255
        if preserve_thin_appendages and not is_perched:
            fg_keep = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        else:
            fg_open = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_keep = cv2.erode(fg_open, np.ones((3, 3), np.uint8)) > 0

        cel_cut = cv2.bilateralFilter(cut_arr, d=11, sigmaColor=50, sigmaSpace=50)
        cel_cut = cv2.bilateralFilter(cel_cut, d=11, sigmaColor=50, sigmaSpace=50)

        stripped = fg_keep & (arr[:, :, 3] < 165)
        painted_px = stripped & (raw_arr.mean(axis=-1) <= 225)
        faded_px = stripped & (raw_arr.mean(axis=-1) > 225)
        arr[painted_px, :3] = raw_arr[painted_px]
        arr[faded_px, :3] = cel_cut[faded_px]
        arr[fg_keep, 3] = 255

    # Strip tiny detached noise specks while keeping main body + perch / thin whip tail segments
    min_comp_frac = 0.002 if preserve_thin_appendages else 0.008
    fg_bin = (arr[:, :, 3] >= 165).astype(np.uint8)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(fg_bin)
    if num_l > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = float(areas.max())
        keep_comp = np.zeros_like(fg_bin, dtype=bool)
        for lbl_idx in range(1, num_l):
            if float(stats[lbl_idx, cv2.CC_STAT_AREA]) >= min_comp_frac * max_area:
                keep_comp |= labels == lbl_idx
        arr[~keep_comp, 3] = 0

    cut_clean = Image.fromarray(arr)
    box = subject_bbox(cut_clean)
    if box is None:
        rgb = raw_im.convert("RGB")
        rgb.save(dst_path)
        return rgb

    subj = cut_clean.crop(box)
    max_dim = max(subj.size)
    fill = max_dim / 1024.0
    if 0.74 <= fill <= 0.82:
        subj_final = subj
    else:
        scale = round(1024 * 0.78) / max_dim
        nw, nh = max(1, round(subj.width * scale)), max(1, round(subj.height * scale))
        subj_final = subj.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    paste_x = (1024 - subj_final.width) // 2
    paste_y = (1024 - subj_final.height) // 2
    canvas.paste(subj_final, (paste_x, paste_y), subj_final)
    rgb = canvas.convert("RGB")
    rgb.save(dst_path)
    return rgb
