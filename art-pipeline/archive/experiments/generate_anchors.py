"""Generate missing-class style anchor candidates via Nano Banana Pro (gemini-3-pro-image).

Uses the 3 top exemplars from STYLE_SPEC.md (american_marten, elk, common_box_turtle)
as frozen style references + a vetted real photo per species for anatomy/markings,
and generates candidates adhering to the 10 rules in STYLE_SPEC.md.

Usage:
    python art-pipeline/generate_anchors.py [--candidates 2] [--species spring_peeper ...]
"""

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "art-pipeline"
MASTERS = PIPELINE / "masters"
OUTDIR = PIPELINE / "anchors_nbp"
REVIEW_DIR = PIPELINE / "style_review"

MODEL = "models/gemini-3-pro-image"

EXEMPLARS = [
    MASTERS / "american_marten.png",
    MASTERS / "elk.png",
    MASTERS / "common_box_turtle.png",
]

ANCHOR_SPECIES = {
    "spring_peeper": {
        "common_name": "Spring Peeper (Pseudacris crucifer)",
        "class_group": "Amphibian (Frog)",
        "photo_path": PIPELINE / "batch_run" / "spring_peeper_photo.png",
        "vignette": (
            "Perched alertly on a single minimal muted-brown twig segment, body in three-quarter view "
            "facing left, all four limbs visible with toe pads, pure solid white background outside "
            "the frog and twig."
        ),
        "diagnostic_notes": (
            "Warm tan-brown to golden-olive skin, signature dark brown X-shaped cross mark on the back, "
            "darker bands on legs, creamy pale belly, large alert dark eye with golden iris and crisp highlight. "
            "Give the frog real painted 3D volume (clear lit top/side and darker shadow underside, never flat)."
        ),
    },
    "american_toad": {
        "common_name": "American Toad (Anaxyrus americanus)",
        "class_group": "Amphibian (Toad)",
        "photo_path": PIPELINE / "vetting" / "funnel_cache" / "291439207.jpg",
        "vignette": (
            "Sitting alertly in a three-quarter posture facing left on invisible ground (no props, "
            "no ground plane, no cast shadow), pure solid white background."
        ),
        "diagnostic_notes": (
            "Rich terracotta-rust, warm brown, and dark umber blotches, prominent kidney-shaped parotoid "
            "glands behind the eyes, warts rendered as flat stylized painted shapes concentrated on the back "
            "and shoulders (not 3D plastic beads), pale cream chest with dark speckles, bright golden-copper iris. "
            "Strong painted light-to-shadow modelling so the body turns in space."
        ),
    },
    "eastern_red-backed_salamander": {
        "common_name": "Eastern Red-backed Salamander (Plethodon cinereus)",
        "class_group": "Amphibian (Salamander)",
        "photo_path": PIPELINE / "vetting" / "funnel_cache" / "273453455.jpg",
        "vignette": (
            "Walking in a natural sinuous S-curve on invisible ground, head turned slightly toward the viewer "
            "facing left, all four small legs and complete tail visible, no props, pure solid white background."
        ),
        "diagnostic_notes": (
            "Rich brick-red to orange-red dorsal stripe running from behind the head down the back, dark slate-black "
            "flanks with subtle pale mottling along the lower sides, costal grooves implied cleanly with a few "
            "stylized strokes (not textbook pen cross-hatching), subtle wet-skin highlight on the eye and nose."
        ),
    },
    "western_fence_lizard": {
        "common_name": "Western Fence Lizard (Sceloporus occidentalis)",
        "class_group": "Small Lizard",
        "photo_path": PIPELINE / "batch_run" / "western_fence_lizard_photo.png",
        "vignette": (
            "Perched in an alert push-up posture on a small, quiet, muted warm-gray stone vignette (stone much "
            "lower contrast and saturation than the lizard), facing left in three-quarter view with head turned "
            "slightly toward the viewer, complete tail and all four legs visible, pure solid white background."
        ),
        "diagnostic_notes": (
            "Warm tan and dark brown wavy dorsal bands, bright turquoise-blue patch visible along the side of the "
            "throat and belly edge, keeled scales indicated as sparse flat stylized marks at the shoulder and back "
            "ridge (never uniform cross-hatching or 3D beads), strong lit top and shadow belly."
        ),
    },
    "gopher_snake": {
        "common_name": "Gopher Snake (Pituophis catenifer)",
        "class_group": "Snake",
        "photo_path": PIPELINE / "vetting" / "funnel_cache" / "517919973.jpg",
        "vignette": (
            "Single adult snake in a living S-coiled resting posture with forebody and head raised alertly "
            "facing left, coils overlapping to create strong painted depth, no props, pure solid white background."
        ),
        "diagnostic_notes": (
            "Warm golden-straw and buff body with bold dark chocolate-brown dorsal saddles and alternating flank "
            "blotches, dark eye-stripe, scales implied by clean painted color bands and sparse stylized edge marks "
            "rather than drawing every individual scale, clear lit top of coils and darker shadow on lower/inner coils."
        ),
    },
    "american_robin": {
        "common_name": "American Robin (Turdus migratorius)",
        "class_group": "Small Bird",
        "photo_path": PIPELINE / "batch_run" / "american_robin_photo.png",
        "vignette": (
            "Standing alertly on invisible ground in a three-quarter view facing left, head turned slightly "
            "toward viewer, both legs and complete tail visible, no props, pure solid white background."
        ),
        "diagnostic_notes": (
            "Rich warm brick-orange breast pushed to its best believable saturation, dark slate-gray back and wings, "
            "near-black head with crisp broken white eye-ring, bright yellow bill with dark tip, white undertail "
            "coverts. Feather groups drawn as calm painted color blocks with concentrated stylized marks at wing "
            "tips and breast edge (not loose artsy brushwork or textbook hatching)."
        ),
    },
    "downy_woodpecker": {
        "common_name": "Downy Woodpecker (Dryobates pubescens)",
        "class_group": "Small Bird",
        "photo_path": PIPELINE / "batch_run" / "downy_woodpecker_photo.png",
        "vignette": (
            "Clinging alertly to a single minimal muted-brown vertical bark/branch segment (branch quieter and "
            "less saturated than the bird), facing left in three-quarter profile, tail braced, pure solid white background."
        ),
        "diagnostic_notes": (
            "Crisp black-and-white pied plumage with warm off-white lights and cool gray shadows on the white breast/back "
            "(never flat pure white), bold white wing spots, crisp black and white facial stripes, bright red nape patch "
            "on the back of the head, short chisel bill, alert dark eye with highlight."
        ),
    },
    "great_egret": {
        "common_name": "Great Egret (Ardea alba)",
        "class_group": "Large Bird (Wader)",
        "photo_path": PIPELINE / "batch_run" / "great_egret_photo.png",
        "vignette": (
            "Wading in a shallow, quiet, stylized pool of muted blue-teal water with simple ripple lines "
            "around the lower legs, body in three-quarter view facing left with neck in a graceful living curve, "
            "pure solid white background outside the bird and water vignette."
        ),
        "diagnostic_notes": (
            "White plumage rendered with real painted color temperature — warm ivory-cream on the lit upper back "
            "and neck, cool blue-gray shadow steps on the underside and far wing (like the snowy owl and mountain goat, "
            "never flat blank white), rich golden-yellow dagger bill, lime-green lores in front of the bright yellow eye, "
            "gloss-free dark slate-black legs."
        ),
    },
    "wood_duck": {
        "common_name": "Wood Duck (Aix sponsa)",
        "class_group": "Large Bird (Waterfowl)",
        "photo_path": PIPELINE / "vetting" / "funnel_cache" / "116041318.jpg",
        "vignette": (
            "Floating on a small, quiet, stylized pool of muted blue water with simple ripple lines, "
            "body in three-quarter view facing left, head turned slightly toward viewer, pure solid white background."
        ),
        "diagnostic_notes": (
            "Male breeding plumage rendered in matte painted color steps (never metallic/glossy AI sheen): "
            "deep emerald-green and plum-purple crest with crisp white stripes, bright red eye and red-orange bill base, "
            "rich chestnut-burgundy breast with tiny stylized white fleck marks, warm buff flanks bordered by a crisp "
            "black-and-white vertical bar."
        ),
    },
}


def load_api_key() -> str:
    tokens_file = ROOT / "Tokens-Local.txt"
    for line in tokens_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("AQ."):
            return line.split("=")[0].strip()
    raise RuntimeError("No AQ.* Gemini API key found in Tokens-Local.txt")


def encode_image_resized(path: Path, max_dim: int = 1024) -> dict:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return {
        "inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        }
    }


def build_prompt(spec: dict) -> str:
    return f"""You are creating a production wildlife illustration for AnimlVision.

Images 1, 2, and 3 (American Marten, Elk, Common Box Turtle) are the FROZEN HOUSE STYLE EXEMPLARS.
Image 4 is the REAL REFERENCE PHOTO of the target species: {spec['common_name']}.

Generate a single illustration of a {spec['common_name']} that matches the exact illustration style of Images 1–3 and the true anatomy, proportions, and diagnostic markings of Image 4.

STRICT HOUSE STYLE RULES (from STYLE_SPEC.md):
1. TEXTURE AS FLAT STYLIZED MARKS: Draw fur, feathers, scales, or skin bumps as flat stylized painted shapes concentrated at focal areas and edges, leaving larger body surfaces as calm painted color. NEVER use uniform pen cross-hatching over the whole body, and NEVER render scales or bumps as individually 3D-lit plastic beads.
2. PAINTED DEPTH: Build 3D form from 2–3 distinct, crisp painted color steps (clear lit side and shadow side, far limbs darker, shadow under belly). NEVER use airbrushed gradients, blurry smudged blends, or glossy/shiny 3D rendering.
3. COLOR PUSHED, NEVER INVENTED: Render the species' true colors and diagnostic markings at their richest believable version with crisp color steps and a small highlight on the eye. Give white or neutral areas real color temperature (warm cream lights, cool gray/slate shadows).
   Species-specific color & markings: {spec['diagnostic_notes']}
4. CLEAN OUTLINE: Continuous thin-to-medium dark outline (dark warm brown or darker shade of adjacent fill, ~2–3px weight) all the way around, matching the marten, elk, and box turtle. Not a thick black sticker outline.
5. LIVING POSE & FRAMING: {spec['vignette']}
6. FORMAT: Square 1:1 aspect ratio, single animal filling roughly 75% of the frame, pure solid white (#FFFFFF) background everywhere outside the animal and its minimal vignette. No text, no borders, no cast ground shadow."""


def generate_one(api_key: str, exemplar_parts: list[dict], slug: str, spec: dict, idx: int, force: bool = False) -> Path:
    species_dir = OUTDIR / slug
    species_dir.mkdir(parents=True, exist_ok=True)
    out_path = species_dir / f"{slug}_cand{idx}.png"
    if out_path.exists() and not force:
        return out_path

    ref_part = encode_image_resized(spec["photo_path"], max_dim=1024)
    prompt_text = build_prompt(spec)

    payload = {
        "contents": [
            {
                "parts": [
                    *exemplar_parts,
                    ref_part,
                    {"text": prompt_text},
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "temperature": 1.0,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={api_key}"
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for p in parts:
        if "inlineData" in p and p["inlineData"].get("data"):
            raw_bytes = base64.b64decode(p["inlineData"]["data"])
            im = Image.open(BytesIO(raw_bytes)).convert("RGB")
            im.save(out_path, format="PNG")
            dt = time.time() - t0
            print(f"[OK] {slug} cand {idx} ({im.size[0]}x{im.size[1]}, {dt:.1f}s) -> {out_path.relative_to(ROOT)}", flush=True)
            return out_path

    raise RuntimeError(f"No image returned for {slug} cand {idx}: {json.dumps(data)[:300]}")


def build_review_sheets(slugs: list[str], num_candidates: int) -> list[Path]:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 17)
        font_sub = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = font_title

    cell_w = 460
    cell_h = 460
    header_h = 48
    cols = 1 + num_candidates

    # Split into sheets of up to 5 species each for crisp readability
    sheet_paths = []
    chunk_size = 5
    for chunk_idx in range(0, len(slugs), chunk_size):
        chunk = slugs[chunk_idx : chunk_idx + chunk_size]
        rows = len(chunk)
        sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + header_h)), "white")
        draw = ImageDraw.Draw(sheet)

        for r, slug in enumerate(chunk):
            spec = ANCHOR_SPECIES[slug]
            y0 = r * (cell_h + header_h)

            # Column 0: Reference photo
            ref_im = Image.open(spec["photo_path"]).convert("RGB")
            ref_im.thumbnail((cell_w - 16, cell_h - 16), Image.LANCZOS)
            sheet.paste(ref_im, ((cell_w - ref_im.width) // 2, y0 + header_h + (cell_h - ref_im.height) // 2))
            draw.text((12, y0 + 12), f"{slug} — [{spec['class_group']}]", fill="#111111", font=font_title)

            # Candidate columns
            for c in range(1, num_candidates + 1):
                cand_p = OUTDIR / slug / f"{slug}_cand{c}.png"
                x0 = c * cell_w
                draw.text((x0 + 12, y0 + 14), f"Candidate {c}", fill="#333333", font=font_sub)
                if cand_p.exists():
                    cim = Image.open(cand_p).convert("RGB")
                    cim.thumbnail((cell_w - 16, cell_h - 16), Image.LANCZOS)
                    sheet.paste(cim, (x0 + (cell_w - cim.width) // 2, y0 + header_h + (cell_h - cim.height) // 2))

            # Subtle row divider
            if r > 0:
                draw.line([(0, y0), (cols * cell_w, y0)], fill="#dddddd", width=2)

        out_sheet = REVIEW_DIR / f"anchors_nbp_sheet_{chunk_idx // chunk_size + 1}.jpg"
        sheet.save(out_sheet, quality=92)
        sheet_paths.append(out_sheet)
        print(f"Saved review sheet: {out_sheet.relative_to(ROOT)}", flush=True)

    return sheet_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*", default=list(ANCHOR_SPECIES.keys()))
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    api_key = load_api_key()
    exemplar_parts = [encode_image_resized(p, max_dim=1024) for p in EXEMPLARS]

    tasks = []
    for slug in args.species:
        if slug not in ANCHOR_SPECIES:
            raise ValueError(f"Unknown species slug: {slug}")
        for c in range(1, args.candidates + 1):
            tasks.append((slug, ANCHOR_SPECIES[slug], c))

    print(f"Generating {len(tasks)} images across {len(args.species)} species using {MODEL}...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(generate_one, api_key, exemplar_parts, slug, spec, c, args.force): (slug, c)
            for slug, spec, c in tasks
        }
        for fut in as_completed(futs):
            slug, c = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERR] {slug} cand {c}: {e}", flush=True)

    build_review_sheets(args.species, args.candidates)


if __name__ == "__main__":
    main()
