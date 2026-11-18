"""Fetch reference-photo candidates for species awaiting art.

Uses a 3-tier aesthetic hierarchy so DSLR telephoto field-guide photos beat
harsh-sunlight cell-phone snapshots:
  Tier 1 — Wikimedia Commons curated/encyclopedia photographs (scientific + common name)
  Tier 2 — iNaturalist hand-curated taxon guide photos (/v1/taxa/{taxon_id})
  Tier 3 — iNaturalist research-grade wild observations (votes + recency fallback)

Usage:
    python art-pipeline/fetch_photo_candidates.py [--batch expansion/test_batch_20.json]
                                                  [--per-species 12]
                                                  [--out art-pipeline/vetting/candidates.json]
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

INAT_OBS = "https://api.inaturalist.org/v1/observations"
INAT_TAXA = "https://api.inaturalist.org/v1/taxa"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "AnimlVisionArtPipeline/1.0 (wildlife reference selector)"}

BAD_TITLE_WORDS = (
    "map", "range", "distribution", "skull", "skeleton", "bone", "track", "footprint",
    "scat", "egg", "nest", "museum", "specimen", "taxidermy", "stamp", "coin", "logo",
    "icon", "drawing", "illustration", "engraving", "plate", "diagram", "chart", "dead",
)


def fetch_wikimedia_commons(scientific_name: str, common_name: str, limit: int = 8) -> list[dict]:
    """Tier 1: High-res DSLR wildlife photos from Wikimedia Commons."""
    out = []
    queries = [f'"{scientific_name}"', f'"{common_name}" {scientific_name}']
    seen_urls = set()

    for q in queries:
        if len(out) >= limit:
            break
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {q}",
            "gsrnamespace": 6,
            "gsrlimit": 15,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 1024,
        }
        try:
            r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=25)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
        except Exception:
            continue

        for page in pages.values():
            title = page.get("title", "").lower()
            if any(w in title for w in BAD_TITLE_WORDS):
                continue
            ii_list = page.get("imageinfo", [])
            if not ii_list:
                continue
            ii = ii_list[0]
            w, h = ii.get("width", 0), ii.get("height", 0)
            if w < 700 or h < 500:
                continue
            orig_url = ii.get("url", "")
            thumb_url = ii.get("thumburl") or orig_url
            if not orig_url or orig_url in seen_urls:
                continue
            if not orig_url.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            ext = ii.get("extmetadata", {})
            lic = (ext.get("LicenseShortName", {}).get("value", "") or "").lower()
            if not any(k in lic for k in ("cc0", "cc by", "cc-by", "public domain", "pd")):
                continue
            artist = ext.get("Artist", {}).get("value", "Wikimedia Commons")
            seen_urls.add(orig_url)
            pid = int(hashlib.md5(orig_url.encode("utf-8")).hexdigest()[:8], 16)
            out.append({
                "photo_id": pid,
                "source_tier": "wikimedia_commons",
                "medium_url": thumb_url,
                "original_url": thumb_url if w > 2400 else orig_url,
                "license": lic,
                "attribution": artist,
                "observation_url": ii.get("descriptionurl", orig_url),
            })
            if len(out) >= limit:
                break
    return out


def fetch_inat_curated_taxon(taxon_id: int, limit: int = 8) -> list[dict]:
    """Tier 2: Hand-curated field-guide taxon photos from iNaturalist /v1/taxa/{id}."""
    out = []
    try:
        r = requests.get(f"{INAT_TAXA}/{taxon_id}", headers=HEADERS, timeout=25)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return out
        for tp in results[0].get("taxon_photos", []):
            p = tp.get("photo", {})
            lic = (p.get("license_code") or "").lower()
            if lic not in ("cc0", "cc-by", "pd"):
                continue
            url = p.get("url") or p.get("medium_url")
            if not url:
                continue
            out.append({
                "photo_id": p["id"],
                "source_tier": "inat_curated_taxon",
                "medium_url": url.replace("square", "medium"),
                "original_url": url.replace("square", "original"),
                "license": lic,
                "attribution": p.get("attribution"),
                "observation_url": f"https://www.inaturalist.org/taxa/{taxon_id}",
            })
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


def fetch_inat_obs(taxon_id: int, order_by: str, n: int) -> list[dict]:
    """Tier 3: Research-grade wild observation photos from iNaturalist."""
    r = requests.get(INAT_OBS, params={
        "taxon_id": taxon_id, "place_id": 1,
        "quality_grade": "research", "captive": "false",
        "photo_license": "cc0,cc-by", "order_by": order_by,
        "per_page": n,
    }, headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = []
    for obs in r.json()["results"]:
        for p in obs.get("photos", [])[:1]:
            url = p["url"]
            out.append({
                "photo_id": p["id"],
                "source_tier": "inat_observation",
                "medium_url": url.replace("square", "medium"),
                "original_url": url.replace("square", "original"),
                "license": p.get("license_code"),
                "attribution": p.get("attribution"),
                "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="expansion/test_batch_20.json")
    ap.add_argument("--per-species", type=int, default=12)
    ap.add_argument("--out", default="art-pipeline/vetting/candidates.json")
    args = ap.parse_args()

    batch = json.loads(Path(args.batch).read_text())
    result = {}
    for sp in batch:
        slug = sp["common_name"].lower().replace(" ", "_")
        seen, cands = set(), []

        # Tier 1: Wikimedia Commons DSLR / encyclopedia photos
        for c in fetch_wikimedia_commons(sp["scientific_name"], sp["common_name"], limit=8):
            if c["photo_id"] not in seen:
                seen.add(c["photo_id"])
                cands.append(c)

        # Tier 2: iNaturalist hand-curated taxon page photos
        for c in fetch_inat_curated_taxon(sp["taxon_id"], limit=8):
            if c["photo_id"] not in seen:
                seen.add(c["photo_id"])
                cands.append(c)

        # Tier 3: iNaturalist observations (votes + recency fallback)
        for order in ("votes", "created_at"):
            for c in fetch_inat_obs(sp["taxon_id"], order, args.per_species):
                if c["photo_id"] not in seen:
                    seen.add(c["photo_id"])
                    cands.append(c)
            time.sleep(0.8)

        result[slug] = {
            "common_name": sp["common_name"],
            "scientific_name": sp["scientific_name"],
            "taxon_id": sp["taxon_id"],
            "candidates": cands[: args.per_species * 2],
        }
        print(f"{slug}: {len(result[slug]['candidates'])} candidates")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
