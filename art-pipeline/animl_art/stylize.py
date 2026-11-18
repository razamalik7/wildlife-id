"""SDXL + MultiControlNet (Canny/Depth) + animl_style_v7 stylization and anatomical conditioning."""

import gc
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_propagation

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

_RAY_FIN_EYE_NEG = (
    "translucent wing, transparent fin, fin rays, bony fish fin rays, "
    "operculum gill cover, giant fish eye, bulging fish eye, "
)

# (keyword_in_common_lower, required_class_or_None, phrase_fn(sci), extra_neg_prefix)
_SUBJECT_SANITIZER_RULES: list[tuple[str, str | None, object, str]] = [
    # --- Mammals ---
    (
        "prairie dog",
        "Mammalia",
        lambda sci: f"a plump unstriped buff-tan Cynomys prairie marmot ({sci})",
        "canine, dog snout, dog ears, puppy, chipmunk stripes, lateral stripes, dorsal stripes, striped back, black and white stripes, pink nose, pink snout, pink feet, hairless rat tail, rat, mouse, ",
    ),
    (
        "mountain beaver",
        "Mammalia",
        lambda sci: f"a stocky burrowing Aplodontia rodent with a stubby hidden tail ({sci})",
        "flat scaly beaver paddle tail, aquatic beaver tail, webbed hind feet, ",
    ),
    (
        "antelope squirrel",
        "Mammalia",
        lambda sci: f"a small striped Ammospermophilus desert ground squirrel ({sci})",
        "antelope horns, hooves, ungulate, deer legs, ",
    ),
    (
        "flying squirrel",
        "Mammalia",
        lambda sci: f"a nocturnal Glaucomys gliding squirrel with furred patagium membrane ({sci})",
        "bird wings, feathers, feathered wings, bird beak, ",
    ),
    (
        "sea lion",
        "Mammalia",
        lambda sci: f"a large golden-brown {sci.split()[0]} eared seal pinniped ({sci})",
        "lion mane, cat face, feline paws, terrestrial mammal legs, ",
    ),
    (
        "kangaroo rat",
        "Mammalia",
        lambda sci: f"a small bipedal Dipodomys desert rodent ({sci})",
        "marsupial kangaroo, pouch, giant kangaroo, ",
    ),
    # --- Birds ---
    (
        "turkey vulture",
        "Aves",
        lambda sci: f"a large dark raptor-like Cathartes New World vulture with a featherless red head ({sci})",
        "domestic turkey fan tail, turkey wattles, caruncle snood, barnyard turkey, ",
    ),
    (
        "catbird",
        "Aves",
        lambda sci: f"a sleek slate-gray passerine songbird ({sci})",
        "domestic cat, cat ears, whiskers, feline fur, four legs, ",
    ),
    (
        "cowbird",
        "Aves",
        lambda sci: f"a compact dark Molothrus icterid songbird ({sci})",
        "cow, bovine horns, cattle, hooves, ",
    ),
    (
        "titmouse",
        "Aves",
        lambda sci: f"a small crested Baeolophus parid songbird ({sci})",
        "mouse ears, rodent tail, whiskers, mouse fur, ",
    ),
    (
        "nighthawk",
        "Aves",
        lambda sci: f"a sleek cryptic-plumaged Chordeiles nightjar bird ({sci})",
        "hooked hawk beak, sharp eagle talons, raptor hawk, ",
    ),
    (
        "frogmouth",
        "Aves",
        lambda sci: f"a cryptic nocturnal caprimulgiform bird with a wide hooked bill ({sci})",
        "amphibian frog, moist skin, webbed frog feet, ",
    ),
    (
        "woodcock",
        "Aves",
        lambda sci: f"a plump short-legged Scolopax shorebird with a very long straight bill ({sci})",
        "rooster plumage, chicken comb, pheasant tail, ",
    ),
    # --- Amphibians ---
    (
        "squirrel tree",
        "Amphibia",
        lambda sci: f"a smooth-skinned arboreal Dryophytes treefrog with sticky toe pads ({sci})",
        "squirrel fur, bushy squirrel tail, rodent claws, whiskers, ",
    ),
    (
        "cricket frog",
        "Amphibia",
        lambda sci: f"a small warty Acris terrestrial chorus frog ({sci})",
        "insect cricket, antennae, six insect legs, insect wings, ",
    ),
    (
        "barking tree",
        "Amphibia",
        lambda sci: f"a stout smooth-skinned Dryophytes treefrog ({sci})",
        "dog, bark wood log, puppy, canine snout, ",
    ),
    (
        "leopard frog",
        "Amphibia",
        lambda sci: f"a sleek Lithobates ranid frog with dark oval dorsal spots ({sci})",
        "feline leopard, fur, mammal ears, tail, whiskers, claws, ",
    ),
    (
        "pickerel frog",
        "Amphibia",
        lambda sci: f"a square-spotted Lithobates ranid frog ({sci})",
        "fish scales, pectoral fins, gill slits, pike fish body, ",
    ),
    (
        "chorus frog",
        "Amphibia",
        lambda sci: f"a small striped Pseudacris hylid frog ({sci})",
        "musical notes, microphone, fur, feathers, ",
    ),
    (
        "gopher frog",
        "Amphibia",
        lambda sci: f"a stout plump Lithobates ranid frog ({sci})",
        "gopher fur, rodent incisors, digging claws, tail, ",
    ),
    (
        "sheep frog",
        "Amphibia",
        lambda sci: f"a small stout microhylid frog ({sci})",
        "woolly sheep, fleece, horns, hooves, ",
    ),
    (
        "pig frog",
        "Amphibia",
        lambda sci: f"a large aquatic Lithobates true frog ({sci})",
        "pig snout, pink pigskin, swine ears, trotters, ",
    ),
    (
        "bullfrog",
        "Amphibia",
        lambda sci: f"a large aquatic Lithobates true frog ({sci})",
        "bull horns, bovine cattle, hooves, tail, ",
    ),
    (
        "spadefoot",
        "Amphibia",
        lambda sci: f"a plump toad-like burrowing pelobatoid amphibian with vertical pupils ({sci})",
        "metal spade, shovel tool, rodent claws, ",
    ),
    # --- Reptiles ---
    (
        "glass lizard",
        "Reptilia",
        lambda sci: f"a sleek completely legless serpentine Ophisaurus lizard with smooth glossy scales and external ear openings ({sci})",
        "legs, four legs, limbs, feet, toes, claws, transparent glass, broken glass, ",
    ),
    (
        "legless lizard",
        "Reptilia",
        lambda sci: f"a sleek completely legless serpentine Anniella lizard ({sci})",
        "legs, four legs, limbs, feet, toes, claws, ",
    ),
    (
        "wormsnake",
        "Reptilia",
        lambda sci: f"a tiny slender glossy Carphophis colubrid snake with smooth iridescent scales ({sci})",
        "earthworm segments, clitellum ring, slimy segmented worm, legs, ",
    ),
    (
        "worm snake",
        "Reptilia",
        lambda sci: f"a tiny slender glossy colubrid snake with smooth scales ({sci})",
        "earthworm segments, clitellum ring, slimy segmented worm, legs, ",
    ),
    (
        "hognose",
        "Reptilia",
        lambda sci: f"a heavy-bodied Heterodon colubrid snake with an upturned keeled snout scale ({sci})",
        "pink pig snout, mammal nose, swine ears, legs, ",
    ),
    (
        "hog-nosed",
        "Reptilia",
        lambda sci: f"a heavy-bodied colubrid snake with an upturned snout scale ({sci})",
        "pink pig snout, mammal nose, swine ears, legs, ",
    ),
    (
        "ratsnake",
        "Reptilia",
        lambda sci: f"a slender semi-arboreal Pantherophis colubrid constrictor snake ({sci})",
        "rodent rat, whiskers, fur, rodent ears, four legs, ",
    ),
    (
        "rat snake",
        "Reptilia",
        lambda sci: f"a slender semi-arboreal colubrid snake ({sci})",
        "rodent rat, whiskers, fur, rodent ears, four legs, ",
    ),
    (
        "milksnake",
        "Reptilia",
        lambda sci: f"a glossy smooth-scaled Lampropeltis tricolor banded colubrid snake ({sci})",
        "milk bottle, dairy cow, spots, legs, ",
    ),
    (
        "milk snake",
        "Reptilia",
        lambda sci: f"a glossy smooth-scaled Lampropeltis colubrid snake ({sci})",
        "milk bottle, dairy cow, legs, ",
    ),
    (
        "kingsnake",
        "Reptilia",
        lambda sci: f"a glossy smooth-scaled Lampropeltis colubrid snake ({sci})",
        "gold crown, king cobra hood, four legs, ",
    ),
    (
        "king snake",
        "Reptilia",
        lambda sci: f"a glossy smooth-scaled Lampropeltis colubrid snake ({sci})",
        "gold crown, king cobra hood, four legs, ",
    ),
    (
        "fox snake",
        "Reptilia",
        lambda sci: f"a blotched Pantherophis colubrid snake ({sci})",
        "fox fur, fox ears, bushy tail, four legs, ",
    ),
    (
        "bullsnake",
        "Reptilia",
        lambda sci: f"a large blotched Pituophis colubrid snake ({sci})",
        "bull horns, cattle hooves, fur, four legs, ",
    ),
    (
        "gopher snake",
        "Reptilia",
        lambda sci: f"a large blotched Pituophis colubrid snake ({sci})",
        "gopher fur, rodent claws, four legs, ",
    ),
    (
        "coachwhip",
        "Reptilia",
        lambda sci: f"a very slender fast Masticophis colubrid snake with braided tail scales ({sci})",
        "leather whip handle, wagon, four legs, ",
    ),
    (
        "alligator lizard",
        "Reptilia",
        lambda sci: f"a slender Elgaria lizard ({sci})",
        "alligator snout, crocodile head, caiman, ",
    ),
    (
        "horned lizard",
        "Reptilia",
        lambda sci: f"a flat-bodied spiny Phrynosoma lizard ({sci})",
        "mammal horns, bull horns, goat horns, ",
    ),
    (
        "leopard lizard",
        "Reptilia",
        lambda sci: f"a long-tailed spotted Gambelia desert lizard ({sci})",
        "feline leopard, fur, mammal ears, whiskers, ",
    ),
    (
        "zebra-tailed lizard",
        "Reptilia",
        lambda sci: f"a slender long-legged Callisaurus desert lizard with black-and-white banded tail underside ({sci})",
        "equine zebra, hooves, horse mane, stripes over body, ",
    ),
    (
        "earless lizard",
        "Reptilia",
        lambda sci: f"a sleek phrynosomatid desert lizard with smooth temporal scales ({sci})",
        "mammal ears, pinnae, fur, ",
    ),
    (
        "chicken turtle",
        "Reptilia",
        lambda sci: f"an elongated-necked aquatic Deirochelys emydid turtle ({sci})",
        "chicken feathers, poultry beak, rooster comb, wings, ",
    ),
    (
        "box turtle",
        "Reptilia",
        lambda sci: f"a high-domed terrestrial Terrapene emydid turtle ({sci})",
        "cardboard box, wooden crate, flat flippered sea turtle, ",
    ),
    (
        "map turtle",
        "Reptilia",
        lambda sci: f"an aquatic Graptemys emydid basking turtle with patterned carapace ({sci})",
        "paper map, cartography grid lines, compass, ",
    ),
    # --- Elasmobranchs ---
    (
        "bat ray",
        "Elasmobranchii",
        lambda sci: f"a smooth-skinned Myliobatis ray ({sci})",
        "bat ears, mammal ears, pointed ears, spots, polka dots, "
        + _RAY_FIN_EYE_NEG
        + "lateral shark gill slits, forked shark caudal fin, ",
    ),
    (
        "cownose ray",
        "Elasmobranchii",
        lambda sci: f"a smooth-winged Rhinoptera eagle ray ({sci})",
        "cow, bovine, mammal snout, horns, "
        + _RAY_FIN_EYE_NEG
        + "lateral shark gill slits, forked shark caudal fin, ",
    ),
    (
        "eagle ray",
        "Elasmobranchii",
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
        "horn shark",
        "Elasmobranchii",
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
        "Elasmobranchii",
        lambda sci: f"a flat-bodied benthic ray-like elasmobranch ({sci})",
        "guitar, musical instrument, strings, " + _RAY_FIN_EYE_NEG,
    ),
]


def is_batoidea_ray_v7(common: str, cls_name: str) -> bool:
    """True for cartilaginous rays, skates, mantas, and guitarfish (avoid reef-fish priors)."""
    return cls_name == "Elasmobranchii" and any(
        k in common.lower() for k in ("ray", "skate", "manta", "guitarfish", "sawfish")
    )


def sanitize_species_subject_v7(common: str, sci: str, cls_name: str, subtype: str) -> tuple[str, str]:
    """Rewrite misleading false-animal common names into unambiguous taxonomic descriptors."""
    c_low = common.lower()
    for kw, req_cls, phrase_fn, extra_neg in _SUBJECT_SANITIZER_RULES:
        if req_cls is not None and cls_name != req_cls:
            continue
        if kw in c_low:
            return phrase_fn(sci), extra_neg
    if subtype == "mammal_aquatic" and cls_name == "Mammalia":
        return (
            f"a {common} marine mammal ({sci}) swimming smoothly in side view",
            "fish scales, vertical fish tail fin, gill slits, bony fish fin rays, terrestrial legs, paws, claws, fur, ",
        )
    if subtype == "mammal_bat" and cls_name == "Mammalia":
        return (
            f"a {common} Chiroptera bat ({sci}) with thin leathery patagium wings",
            "feathered bird wings, bird feathers, avian beak, ",
        )
    if cls_name == "Elasmobranchii" and any(k in c_low for k in ("stingray", "manta", "devil ray", "ray", "skate")):
        return (
            f"a {common} ({sci})",
            _RAY_FIN_EYE_NEG + "lateral shark gill slits, forked shark caudal fin, mammal fur, legs, ",
        )
    return f"a {common} ({sci})", ""


def strip_head_eye_clauses(raw_traits: str) -> str:
    """Strip conflicting eye/head/labial phrases when using `unet_only` LoRA weights."""
    clauses = [c.strip() for c in raw_traits.split(",") if c.strip()]
    kept = []
    for c in clauses:
        low = c.lower()
        if any(w in low for w in ("pupil", "eye", "head", "labial")):
            continue
        kept.append(c)
    return ", ".join(kept)


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

    if any(tok in traits_low for tok in ("collar", "band", "white", "cream", "stripe", "stripes", "marbled", "spotted")):
        neg_base = neg_base.replace("stark white fur streaks, ", "specular glare, ")

    extra_parts: list[str] = []
    if extra_neg:
        extra_parts.append(extra_neg.strip(", "))

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

    if "unstriped" in traits_low or "unstriped" in subj_phrase.lower():
        extra_parts.append("chipmunk stripes, lateral stripes, striped back")

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


def build_canny_from_repaired(im: Image.Image) -> Image.Image:
    """Compute bilateral-smoothed adaptive Canny + outer silhouette edges from a white-bg cutout."""
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

    rgb_edges = np.zeros_like(sm_eq)
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
        and any(
            tok in traits_low
            for tok in ("white underbelly", "white ventral", "white belly", "white underside", "pale underside", "cream underside")
        )
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


def encode_prompt_sdxl_multichunk(pipe, prompt: str, negative_prompt: str, max_chunks: int = 3):
    """225-token Multi-Chunk SDXL prompt encoder (prevents 77-token truncation)."""
    import torch

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
        return torch.cat([h1, h2], dim=-1).to(dtype=dtype), pooled.to(dtype=dtype)

    with torch.no_grad():
        pos_embeds, pos_pooled = _encode_pair(prompt, req_chunks)
        neg_embeds, neg_pooled = _encode_pair(negative_prompt, req_chunks)
    return pos_embeds, neg_embeds, pos_pooled, neg_pooled


def compute_depth_maps(changed_stems: list[str], cut_final_dir: Path, depth_dir: Path) -> None:
    """Compute foreground-normalized Depth-Anything-V2 maps on GPU/CPU and unload before SDXL."""
    if not changed_stems:
        return
    import torch
    from transformers import pipeline as hf_pipeline

    print(f"Computing Depth-Anything-V2 maps for {len(changed_stems)} species...", flush=True)
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1,
    )
    for stem in changed_stems:
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
        Image.fromarray(np.stack([d_norm.astype(np.uint8)] * 3, axis=-1)).save(depth_dir / f"{stem}.png")

    del depth_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sdxl_stylization(
    need_render: list[dict],
    meta: dict,
    raw_dir: Path,
    lora_output_dir: Path,
    sdxl_base_path: Path,
    unet_only_stems: set[str],
) -> None:
    """Run SDXL + MultiControlNet (Canny/Depth) + animl_style_v7-step00002000 over `need_render`."""
    if not need_render:
        return
    import torch
    from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionXLControlNetImg2ImgPipeline
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

    print("Loading SDXL + Canny/Depth MultiControlNet + animl_style_v7-step00002000...", flush=True)
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
        str(sdxl_base_path),
        controlnet=MultiControlNetModel([cnet_canny, cnet_depth]),
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    def _is_auto_unet(it: dict) -> bool:
        return (it["stem"] in unet_only_stems) or is_batoidea_ray_v7(it["common"], it["cls_name"])

    full_items = [it for it in need_render if not _is_auto_unet(it)]
    unet_items = [it for it in need_render if _is_auto_unet(it)]

    for pass_label, weight_name, pass_items in [
        ("full v7-2000", "animl_style_v7-step00002000.safetensors", full_items),
        ("unet_only v7-2000", "animl_style_v7_unet_only.safetensors", unet_items),
    ]:
        if not pass_items:
            continue
        pipe.unload_lora_weights()
        print(f"\n[{pass_label}] Loading {weight_name} for {len(pass_items)} species...", flush=True)
        pipe.load_lora_weights(str(lora_output_dir), weight_name=weight_name)
        for it in pass_items:
            stem = it["stem"]
            m = meta[stem]
            traits = m["visual_traits"]
            is_ray = is_batoidea_ray_v7(it["common"], it["cls_name"])
            if stem in unet_only_stems or is_ray:
                traits = strip_head_eye_clauses(traits)
            prompt, neg_prompt, strength = build_prompt_and_neg_v7(
                it["common"],
                it["sci"],
                it["cls_name"],
                it["subtype"],
                traits,
                m.get("eye_color_in_photo", ""),
                it["is_perched"],
            )
            canny_scale = (
                0.54
                if it["subtype"] == "shark_ray"
                else (0.48 if any(k in stem for k in ("skink", "salamander", "killdeer", "hawk")) else 0.45)
            )
            print(
                f"  Rendering {stem} ({it['common']}) | perched={it['is_perched']} "
                f"strength={strength:.2f} canny={canny_scale:.2f}",
                flush=True,
            )
            cut_im = Image.open(it["cut_p"]).convert("RGB")
            canny_im = Image.open(it["canny_p"]).convert("RGB")
            depth_im = Image.open(it["depth_p"]).convert("RGB")
            cond_cut_im, cond_canny_im = prepare_conditioned_cutout_and_canny_v7(
                cut_im, it["cls_name"], traits, base_canny=canny_im
            )

            pos_e, neg_e, pos_p, neg_p = encode_prompt_sdxl_multichunk(pipe, prompt, neg_prompt, max_chunks=3)
            gen = torch.Generator(device="cuda").manual_seed(42)
            out_im = pipe(
                prompt_embeds=pos_e,
                negative_prompt_embeds=neg_e,
                pooled_prompt_embeds=pos_p,
                negative_pooled_prompt_embeds=neg_p,
                image=cond_cut_im,
                control_image=[cond_canny_im, depth_im],
                strength=strength,
                controlnet_conditioning_scale=[canny_scale, 0.25],
                control_guidance_start=0.0,
                control_guidance_end=[0.76, 0.60],
                guidance_scale=6.0,
                num_inference_steps=36,
                generator=gen,
            ).images[0]
            out_im.save(raw_dir / f"{stem}.png")

    del pipe, cnet_canny, cnet_depth
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Unloaded SDXL + MultiControlNet from VRAM.", flush=True)
