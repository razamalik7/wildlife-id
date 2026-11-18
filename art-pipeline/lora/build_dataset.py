"""Build the SDXL style-LoRA v5 training dataset from the 26 locked masters.

Training data = 26 curated illustrations (11 Core + 15 New Anchors including common_watersnake, all prop-free)
+ automated local gemma3:27b + Wikipedia Description variable-trait captions with 100% inference parity.
Balances dataset repeats:
- dataset/img_4x (num_repeats = 4): 22 single-image species
- dataset/img_2x (num_repeats = 2): 4 multi-pose images (common_box_turtle_3/4, gopher_snake_1/2)
Every species gets exactly 4 repeats per epoch (96 samples = 48 steps/epoch at batch_size=2).

Usage: python art-pipeline/lora/build_dataset.py
"""

import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "art-pipeline"))
from normalize import subject_bbox  # noqa: E402

OLLAMA = "http://localhost:11434"
MASTERS_CORE = REPO / "art-pipeline" / "masters"
MASTERS_NEW = REPO / "art-pipeline" / "masters_new_anchors"
DATASET_DIR = Path(__file__).parent / "dataset"
OUT_6X = DATASET_DIR / "img_6x"
OUT_4X = DATASET_DIR / "img_4x"
OUT_2X = DATASET_DIR / "img_2x"
CACHE_JSON = DATASET_DIR / "v5_gemma27b_captions.json"
SIZE = 1024
FILL = 0.76

STYLE_CLAUSE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink fur and contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

MAMMAL_FUR_ANCHORS = {"american_marten", "rhesus_macaque"}

# Explicit 26-piece v5 training manifest:
# (slug, png_path, wiki_title, is_multi_pose)
V5_MANIFEST: list[tuple[str, Path, str, bool]] = [
    # 11 Core Masters
    ("american_marten", MASTERS_CORE / "american_marten.png", "American marten", False),
    ("elk", MASTERS_CORE / "elk.png", "Elk", False),
    ("caribou", MASTERS_CORE / "caribou.png", "Reindeer", False),
    ("white-tailed_deer", MASTERS_CORE / "white-tailed_deer.png", "White-tailed deer", False),
    ("axis_deer", MASTERS_CORE / "axis_deer.png", "Chital", False),
    ("mountain_goat", MASTERS_CORE / "mountain_goat.png", "Mountain goat", False),
    ("nine-banded_armadillo", MASTERS_CORE / "nine-banded_armadillo.png", "Nine-banded armadillo", False),
    ("harbor_seal", MASTERS_CORE / "harbor_seal.png", "Harbor seal", False),
    ("rhesus_macaque", MASTERS_CORE / "rhesus_macaque.png", "Rhesus macaque", False),
    ("snowy_owl", MASTERS_CORE / "snowy_owl.png", "Snowy owl", False),
    ("desert_tortoise", MASTERS_CORE / "desert_tortoise.png", "Desert tortoise", False),
    # 15 New Anchors (including newly promoted common_watersnake)
    ("american_robin", MASTERS_NEW / "american_robin.png", "American robin", False),
    ("downy_woodpecker", MASTERS_NEW / "downy_woodpecker.png", "Downy woodpecker", False),
    ("wood_duck", MASTERS_NEW / "wood_duck.png", "Wood duck", False),
    ("great_egret", MASTERS_NEW / "great_egret.png", "Great egret", False),
    ("barking_treefrog", MASTERS_NEW / "barking_treefrog.png", "Barking tree frog", False),
    ("spring_peeper", MASTERS_NEW / "spring_peeper.png", "Spring peeper", False),
    ("american_toad", MASTERS_NEW / "american_toad.png", "American toad", False),
    ("eastern_red-backed_salamander", MASTERS_NEW / "eastern_red-backed_salamander.png", "Red-backed salamander", False),
    ("common_box_turtle_3", MASTERS_NEW / "common_box_turtle_3.png", "Common box turtle", True),
    ("common_box_turtle_4", MASTERS_NEW / "common_box_turtle_4.png", "Common box turtle", True),
    ("gopher_snake_1", MASTERS_NEW / "gopher_snake_1.png", "Gopher snake", True),
    ("gopher_snake_2", MASTERS_NEW / "gopher_snake_2.png", "Gopher snake", True),
    ("western_fence_lizard", MASTERS_NEW / "western_fence_lizard.png", "Western fence lizard", False),
    ("eastern_fence_lizard", MASTERS_NEW / "eastern_fence_lizard.png", "Eastern fence lizard", False),
]

MASTER_PROMPT_TEMPLATE = """You are a wildlife illustrator cataloging the visible colors and anatomical markings of a {common_name} illustration alongside its Wikipedia Description.

Wikipedia Description for {common_name}:
\"\"\"{wiki_text}\"\"\"

Your job is to return ONLY one field in JSON: `visible_color_and_markings`.
- RULE 1 (BASE COLOR FROM IMAGE FIRST): Look directly at the attached image first to determine the actual base color of the animal's shell, fur, scales, skin, or plumage. When there are age, sex, or regional variants listed in the Wikipedia description, determine WHICH variant matches the image and use THAT variant's colors. Never use hyphenated green compounds like 'greenish-brown' if the shell/coat is brown or tan.
- RULE 2 (DETAILED MARKINGS FROM WIKIPEDIA + IMAGE): Combine those true base colors with the species' diagnostic visible markings from Wikipedia (e.g., for Common Box Turtle: `'high-domed dark brown carapace with bright orange-yellow radiating markings, orange-brown head and sturdy legs'`; for Gopher Snake: `'sandy tan body with dark brown dorsal blotches and smaller dark lateral spots'`).
- Write a 16-24 word comma-separated list of `[color/pattern] [body part]` phrases for ONLY the parts of the animal visible from this angle.
- DO NOT mention hidden underside parts (no 'plastron', 'ventral scales', 'underwing coverts').
- DO NOT describe pose, camera angle, or art style (no 'gouache', 'painted', 'illustration').
- DO NOT mention environment, water, grass, rocks, or logs.
- DO NOT use the word 'pinkish'.

Return STRICT JSON only:
{{
  "visible_color_and_markings": "16-24 word comma-separated list of visible `[color/pattern] [body part]` phrases matching the variant in the image"
}}"""


def fetch_full_wiki_description(title: str, session: requests.Session) -> str:
    r = session.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1, "format": "json"},
        timeout=20,
    ).json()
    page = list(r["query"]["pages"].values())[0].get("extract", "")
    lead = page.split("==")[0].strip()
    sections = re.findall(
        r"==+\s*(?:Description|Appearance|Physical description|Coloration|Plumage|Characteristics).*?==+(.+?)(?===+\s*\w|$)",
        page,
        re.S | re.I,
    )
    combined = lead + "\n\n" + "\n".join(sections)
    return re.sub(r"\s+", " ", combined)[:3000]


def call_gemma27b_for_master(img_path: Path, common_name: str, wiki_text: str) -> str:
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((768, 768), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = MASTER_PROMPT_TEMPLATE.format(common_name=common_name, wiki_text=wiki_text)
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": "gemma3:27b",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        },
        timeout=180,
    )
    r.raise_for_status()
    data = json.loads(r.json()["message"]["content"])
    return data.get("visible_color_and_markings", "").strip()


def clean_species_name(slug: str) -> str:
    clean = re.sub(r"_\d+$", "", slug)
    return clean.replace("_", " ")


def build_caption(slug: str, visible_colors: str) -> str:
    name = clean_species_name(slug)
    article = "an" if name[0].lower() in "aeiou" else "a"
    return (
        f"animl_gouache_style, {article} {name}, full body wildlife illustration, "
        f"single animal, {visible_colors}, {STYLE_CLAUSE}"
    )


def main() -> None:
    for d in (OUT_6X, OUT_4X, OUT_2X):
        d.mkdir(parents=True, exist_ok=True)
        for old_file in d.iterdir():
            if old_file.is_file():
                old_file.unlink()

    cached_traits: dict[str, str] = {}
    if CACHE_JSON.exists():
        try:
            cached_traits = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
        except Exception:
            cached_traits = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "AnimlVisionBot/1.0"})

    n = 0
    used_ollama = False
    for slug, png, wiki_title, is_multi_pose in V5_MANIFEST:
        if not png.exists():
            raise FileNotFoundError(f"Missing master in v5 manifest: {png}")
        if slug in MAMMAL_FUR_ANCHORS:
            out_dir = OUT_6X
            rep_str = "6x"
        elif is_multi_pose:
            out_dir = OUT_2X
            rep_str = "2x"
        else:
            out_dir = OUT_4X
            rep_str = "4x"
        im = Image.open(png).convert("RGBA")
        box = subject_bbox(im)
        if box is None:
            raise ValueError(f"Empty alpha bbox for {png}")
        subject = im.crop(box)
        target = int(SIZE * FILL)
        scale = target / max(subject.size)
        nw, nh = (max(1, round(d * scale)) for d in subject.size)
        subject = subject.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
        canvas.paste(subject, ((SIZE - nw) // 2, (SIZE - nh) // 2), subject)
        dst_png = out_dir / f"{slug}.png"
        canvas.convert("RGB").save(dst_png)

        if slug in cached_traits and cached_traits[slug]:
            traits = cached_traits[slug]
        else:
            used_ollama = True
            common_name = clean_species_name(slug).title()
            wiki_txt = fetch_full_wiki_description(wiki_title, session)
            traits = call_gemma27b_for_master(dst_png, common_name, wiki_txt)
            cached_traits[slug] = traits
            CACHE_JSON.write_text(json.dumps(cached_traits, indent=2), encoding="utf-8")

        cap = build_caption(slug, traits)
        (out_dir / f"{slug}.txt").write_text(cap, encoding="utf-8")
        print(f"  [{n+1:02d}/26] ({rep_str}) {slug}.png -> '{cap}'", flush=True)
        n += 1

    if used_ollama:
        print("Unloading gemma3:27b from Ollama VRAM...", flush=True)
        requests.post(f"{OLLAMA}/api/generate", json={"model": "gemma3:27b", "keep_alive": 0}, timeout=15)
        time.sleep(2)

    print(f"v6 dataset ready: {n} image/caption pairs across {OUT_6X} (6x), {OUT_4X} (4x), and {OUT_2X} (2x)", flush=True)


if __name__ == "__main__":
    main()
