"""Fetch Wikipedia 'Description' / lead visual traits for our 25 training masters
and format them into standardized, parity-matched captions/prompts.
"""

import json
import re
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parents[1]

V4_MASTERS = [
    ("american_marten", "American marten", "standing three-quarter view"),
    ("elk", "Elk", "standing side three-quarter view"),
    ("caribou", "Reindeer", "standing three-quarter view"),
    ("white-tailed_deer", "White-tailed deer", "standing three-quarter view"),
    ("axis_deer", "Chital", "standing three-quarter view"),
    ("mountain_goat", "Mountain goat", "standing three-quarter view"),
    ("nine-banded_armadillo", "Nine-banded armadillo", "walking side three-quarter view"),
    ("harbor_seal", "Harbor seal", "resting side three-quarter view"),
    ("rhesus_macaque", "Rhesus macaque", "sitting three-quarter view"),
    ("snowy_owl", "Snowy owl", "standing front three-quarter view"),
    ("desert_tortoise", "Desert tortoise", "walking three-quarter view"),
    ("american_robin", "American robin", "standing side three-quarter view"),
    ("downy_woodpecker", "Downy woodpecker", "vertical clinging side profile view"),
    ("wood_duck", "Wood duck", "standing side three-quarter view"),
    ("great_egret", "Great egret", "standing side three-quarter view"),
    ("barking_treefrog", "Barking tree frog", "crouching three-quarter view"),
    ("spring_peeper", "Spring peeper", "crouching three-quarter view"),
    ("american_toad", "American toad", "crouching three-quarter view"),
    ("eastern_red-backed_salamander", "Red-backed salamander", "crawling S-curve top-three-quarter view"),
    ("common_box_turtle_3", "Common box turtle", "walking three-quarter view"),
    ("common_box_turtle_4", "Common box turtle", "standing front-three-quarter view"),
    ("gopher_snake_1", "Gopher snake", "coiled S-curve three-quarter view"),
    ("gopher_snake_2", "Gopher snake", "loosely coiled three-quarter view"),
    ("western_fence_lizard", "Western fence lizard", "crouching three-quarter view"),
    ("eastern_fence_lizard", "Eastern fence lizard", "alert crouching three-quarter view"),
]

COLOR_WORDS = {
    "brown", "black", "white", "grey", "gray", "red", "reddish", "orange", "yellow",
    "green", "olive", "tan", "buff", "chestnut", "rust", "rusty", "rufous", "golden",
    "cream", "pink", "blue", "iridescent", "purple", "bronze", "dark", "pale", "light",
    "bright", "spotted", "spots", "striped", "stripes", "banded", "bands", "blotches",
    "mottled", "speckled", "barred", "bars", "ringed",
    "darker", "lighter", "paler", "blacker", "whiter", "whitish", "yellowish", "brownish",
    "grayish", "greyish", "blackish", "bluish", "greenish", "grizzled", "silver", "silvery",
    "slate", "charcoal", "ivory", "tawny", "sandy", "cinnamon", "ochre", "unstriped",
    "plain", "solid",
}

ANATOMY_WORDS = {
    "patch", "throat", "breast", "belly", "underparts", "upperparts", "back", "dorsal",
    "flank", "flanks", "rump", "tail", "head", "crown", "crest", "bill", "beak", "eye",
    "eyes", "ears", "antlers", "horns", "hooves", "legs", "feet", "webbed", "claws",
    "fur", "coat", "pelage", "plumage", "feathers", "scales", "scutes", "carapace",
    "shell", "skin", "warts", "snout", "muzzle", "face", "neck", "mane", "chest",
    "body", "wings", "armor", "collar", "stripe", "stripes", "spots",
    "coats", "hairs", "hair", "tails", "cheeks", "cheek", "eyebrow", "eyebrows",
    "shoulder", "shoulders", "side", "sides", "limb", "limbs", "fin", "fins",
    "flipper", "flippers", "tusk", "tusks", "whisker", "whiskers", "bellies", "collars",
    "band", "bands", "ring", "rings", "mask", "masks", "mark", "marks", "marking",
    "markings", "plates", "toe", "toes", "wingtip", "wingtips", "color", "coloration",
}

STOP_WORDS = {
    "conservation", "iucn", "population", "habitat", "diet", "predator", "prey",
    "breeding", "reproduction", "egg", "eggs", "nest", "migration", "taxonomy",
    "linnaeus", "subspecies", "fossil", "pleistocene", "hunting", "disease",
    "parasite", "gestation", "lifespan", "vocal", "call", "song", "territor",
    "range", "native", "fledging", "juvenile", "immature", "chick", "larva",
    "tadpole", "winter",
}


def fetch_wiki_batch(titles: list[str]) -> dict[str, str]:
    url = "https://en.wikipedia.org/w/api.php"
    out = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "AnimlVisionBot/1.0 (https://github.com/animl-vision; contact@animl.org)"})
    for t in titles:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": t,
            "redirects": 1,
            "format": "json",
        }
        r = session.get(url, params=params, timeout=20)
        if r.status_code == 200 and r.text.strip().startswith("{"):
            pages = r.json().get("query", {}).get("pages", {})
            for p in pages.values():
                out[t] = p.get("extract", "")
    return out


def _score_visual_sentences(pool_text: str, seen: set[str]) -> list[tuple[int, float, str]]:
    clean = re.sub(r"\([^)]*\)", "", pool_text)
    clean = re.sub(r"\b\d+(?:\.\d+)?\s*(?:to|–|-)\s*\d+(?:\.\d+)?\s*(?:cm|mm|m|in|ft|kg|g|lb|oz)\b", "", clean)
    clean = re.sub(r"\s+", " ", clean)

    scored = []
    for raw_s in re.split(r"(?<=[.!?])\s+", clean):
        s = raw_s.strip()
        if len(s) < 30 or len(s) > 250 or s.lower() in seen:
            continue
        words = re.findall(r"[a-z]+", s.lower())
        c_hits = sum(1 for w in words if w in COLOR_WORDS)
        a_hits = sum(1 for w in words if w in ANATOMY_WORDS)
        if any(sw in s.lower() for sw in STOP_WORDS) and (c_hits < 2 or a_hits < 2):
            continue
        if c_hits >= 1 and a_hits >= 1 and (c_hits + a_hits) >= 3:
            seen.add(s.lower())
            scored.append((c_hits + a_hits, (c_hits + a_hits) / max(len(words), 1), s))
    return scored


def extract_visual_clauses(wiki_text: str) -> str:
    """Extract the top visual clauses (color + anatomy) from the Wikipedia Description/lead."""
    desc_matches = re.findall(
        r"==\s*(?:Description|Appearance|Physical description|Coloration|Plumage|Characteristics)\s*==(.+?)(?=\n==\s*[^=]|$)",
        wiki_text,
        re.S | re.I,
    )
    lead = wiki_text.split("==")[0]
    pool_text = ("\n".join(desc_matches) if desc_matches else "") + "\n" + lead

    seen: set[str] = set()
    scored = _score_visual_sentences(pool_text, seen)
    if len(scored) < 2 and wiki_text:
        body_text = re.split(
            r"\n==\s*(?:References|External links|Further reading|Notes|See also|Explanatory notes)\s*==",
            wiki_text,
            maxsplit=1,
            flags=re.I,
        )[0]
        scored.extend(_score_visual_sentences(body_text, seen))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return " | ".join(x[2] for x in scored[:2])


def main() -> None:
    unique_titles = sorted(set(t for _, t, _ in V4_MASTERS))
    texts = fetch_wiki_batch(unique_titles)

    results = {}
    for slug, title, pose in V4_MASTERS:
        txt = texts.get(title, "")
        vis = extract_visual_clauses(txt)
        results[slug] = {
            "wiki_title": title,
            "pose": pose,
            "wiki_visual_extract": vis,
        }
        print(f"{slug:30s} -> {vis[:155]}")

    (REPO / "art-pipeline" / "wiki_masters_raw.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
