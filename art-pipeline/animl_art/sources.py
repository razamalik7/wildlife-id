"""Reference-photo and Wikipedia trait acquisition from iNaturalist, Wikimedia Commons, and Wikipedia."""

import re
from io import BytesIO

import requests
from PIL import Image

HEADERS = {"User-Agent": "AnimlVisionArtPipeline/2.0 (https://animl.org; contact@animl.org)"}

_GENERAL_BAD_TITLE_WORDS = (
    "map",
    "range",
    "skull",
    "skeleton",
    "egg",
    "track",
    "footprint",
    "stamp",
    ".svg",
    ".webm",
    ".ogg",
    ".pdf",
)

_MARINE_BAD_TITLE_WORDS = (
    "aquarium",
    "zoo",
    "jaws",
    "tooth",
    "teeth",
    "skeleton",
    "skull",
    "caught",
    "fisherman",
    "market",
    "dead",
    "beach",
    "map",
    "range",
    "stamp",
    ".svg",
    ".webm",
    ".ogg",
    ".pdf",
)

COLOR_WORDS = {
    "brown", "black", "white", "grey", "gray", "red", "reddish", "orange", "yellow",
    "green", "olive", "tan", "buff", "chestnut", "rust", "rusty", "rufous", "golden",
    "cream", "pink", "blue", "iridescent", "purple", "bronze", "dark", "pale", "light",
    "bright", "spotted", "spots", "striped", "stripes", "banded", "bands", "blotches",
    "mottled", "speckled", "barred", "bars", "ringed", "darker", "lighter", "paler",
    "blacker", "whiter", "whitish", "yellowish", "brownish", "grayish", "greyish",
    "blackish", "bluish", "greenish", "grizzled", "silver", "silvery", "slate",
    "charcoal", "ivory", "tawny", "sandy", "cinnamon", "ochre", "unstriped", "plain",
    "solid",
}

ANATOMY_WORDS = {
    "patch", "throat", "breast", "belly", "underparts", "upperparts", "back", "dorsal",
    "flank", "flanks", "rump", "tail", "head", "crown", "crest", "bill", "beak", "eye",
    "eyes", "ears", "antlers", "horns", "hooves", "legs", "feet", "webbed", "claws",
    "fur", "coat", "pelage", "plumage", "feathers", "scales", "scutes", "carapace",
    "shell", "skin", "warts", "snout", "muzzle", "face", "neck", "mane", "chest",
    "body", "wings", "armor", "collar", "stripe", "stripes", "spots", "coats", "hairs",
    "hair", "tails", "cheeks", "cheek", "eyebrow", "eyebrows", "shoulder", "shoulders",
    "side", "sides", "limb", "limbs", "fin", "fins", "flipper", "flippers", "tusk",
    "tusks", "whisker", "whiskers", "bellies", "collars", "band", "bands", "ring",
    "rings", "mask", "masks", "mark", "marks", "marking", "markings", "plates", "toe",
    "toes", "wingtip", "wingtips", "color", "coloration",
}

STOP_WORDS = {
    "conservation", "iucn", "population", "habitat", "diet", "predator", "prey",
    "breeding", "reproduction", "egg", "eggs", "nest", "migration", "taxonomy",
    "linnaeus", "subspecies", "fossil", "pleistocene", "hunting", "disease",
    "parasite", "gestation", "lifespan", "vocal", "call", "song", "territor",
    "range", "native", "fledging", "juvenile", "immature", "chick", "larva",
    "tadpole", "winter",
}


def download_pil_compliant(url: str, timeout: int = 20) -> Image.Image | None:
    """Download an image URL with compliant User-Agent and return RGB PIL image."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def _search_commons(queries: tuple[str, ...], bad_words: tuple[str, ...], limit: int) -> list[dict]:
    """Shared Wikimedia Commons namespace-6 search helper."""
    out: list[dict] = []
    seen: set[str] = set()
    for q in queries:
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
                headers=HEADERS,
                timeout=15,
            ).json()
            for pid, page in r.get("query", {}).get("pages", {}).items():
                if pid in seen or "imageinfo" not in page:
                    continue
                t_low = page["title"].lower()
                if any(bw in t_low for bw in bad_words):
                    continue
                seen.add(pid)
                ii = page["imageinfo"][0]
                u = ii.get("thumburl") or ii.get("url", "")
                if u:
                    out.append(
                        {
                            "photo_id": f"commons_{pid}",
                            "original_url": u,
                            "medium_url": u,
                            "title": page["title"],
                        }
                    )
        except Exception:
            pass
    return out


def fetch_commons_files(sci: str, common: str, limit: int = 10) -> list[dict]:
    """Query Wikimedia Commons File namespace (`gsrnamespace=6`) for wild animal photos."""
    return _search_commons((sci, f"{sci} {common}"), _GENERAL_BAD_TITLE_WORDS, limit=limit)


def fetch_inat_curated_taxon(taxon_id: int, limit: int = 8) -> list[dict]:
    """Fetch CC0/CC-BY curated taxon photos from iNaturalist `/v1/taxa/{taxon_id}`."""
    out: list[dict] = []
    try:
        r = requests.get(f"https://api.inaturalist.org/v1/taxa/{taxon_id}", headers=HEADERS, timeout=25)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return out
        for tp in results[0].get("taxon_photos", []):
            p = tp.get("photo", {})
            lic = (p.get("license_code") or "").lower()
            if lic not in ("cc0", "cc-by", "pd", ""):
                continue
            url = p.get("url") or p.get("medium_url")
            if not url:
                continue
            out.append(
                {
                    "photo_id": f"inat_t_{p['id']}",
                    "source_tier": "inat_curated_taxon",
                    "medium_url": url.replace("square", "medium"),
                    "original_url": url.replace("square", "original"),
                    "license": lic,
                    "attribution": p.get("attribution"),
                }
            )
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


def fetch_inat_obs(taxon_id: int, order_by: str = "votes", n: int = 12, place_id: int | None = 1) -> list[dict]:
    """Fetch research-grade wild observation photos from iNaturalist `/v1/observations`."""
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "captive": "false",
        "photo_license": "cc0,cc-by",
        "order_by": order_by,
        "per_page": n,
    }
    if place_id is not None:
        params["place_id"] = place_id
    r = requests.get("https://api.inaturalist.org/v1/observations", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    out: list[dict] = []
    for obs in r.json().get("results", []):
        for p in obs.get("photos", [])[:1]:
            url = p.get("url") or ""
            if url:
                out.append(
                    {
                        "photo_id": f"inat_{p['id']}",
                        "source_tier": "inat_observation",
                        "medium_url": url.replace("square", "medium"),
                        "original_url": url.replace("square", "original"),
                        "license": p.get("license_code"),
                        "attribution": p.get("attribution"),
                    }
                )
    return out


def fetch_marine_candidates_v7(taxon_id: int, sci: str, common: str) -> list[dict]:
    """Fetch & interleave global underwater scuba observations and Commons media for marine taxa."""
    src_taxa = fetch_inat_curated_taxon(taxon_id, limit=10)
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
            headers=HEADERS,
            timeout=25,
        ).json()
        src_obs = [
            {
                "photo_id": f"inat_{p['id']}",
                "medium_url": u.replace("square", "medium"),
                "original_url": u.replace("square", "original"),
            }
            for obs in r.get("results", [])
            for p in obs.get("photos", [])[:1]
            if (u := p.get("url") or "")
        ]
    except Exception:
        src_obs = []

    src_commons = _search_commons(
        (f'"{sci}" underwater', f'"{common}" underwater', sci),
        _MARINE_BAD_TITLE_WORDS,
        limit=10,
    )

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


def fetch_wiki_batch(titles: list[str]) -> dict[str, str]:
    """Fetch plain-text Wikipedia extracts for a list of page titles."""
    url = "https://en.wikipedia.org/w/api.php"
    out: dict[str, str] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "AnimlVisionBot/1.0 (https://github.com/animl-vision; contact@animl.org)"})
    for t in titles:
        try:
            r = session.get(
                url,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": t,
                    "redirects": 1,
                    "format": "json",
                },
                timeout=20,
            )
            if r.status_code == 200 and r.text.strip().startswith("{"):
                pages = r.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    out[t] = p.get("extract", "")
        except Exception:
            pass
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
    """Extract top color + anatomy sentences from Wikipedia Description/lead with body fallback."""
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


def fetch_wiki_clauses_map(species_list: list[tuple[str, str]]) -> dict[str, str]:
    """Fetch Wikipedia visual clauses for `[(common, sci), ...]`, falling back to `sci` on disambiguation/stubs."""
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
            out[common] = extract_visual_clauses(raw_sci.get(sci, ""))
    return out
