"""Fetch reference-photo candidates for species awaiting art.

Pulls up to N CC0/CC-BY research-grade wild observation photos per species
from iNaturalist (vote-ordered plus a recency slice for variety), recording
url, license, attribution, and source observation. Feeds both the vetting
UI (art-pipeline/vetting/) and automated contact sheets.

Usage:
    python art-pipeline/fetch_photo_candidates.py [--batch expansion/test_batch_20.json]
                                                  [--per-species 12]
                                                  [--out art-pipeline/vetting/candidates.json]
"""

import argparse
import json
import time
from pathlib import Path

import requests

INAT = "https://api.inaturalist.org/v1/observations"


def fetch(taxon_id: int, order_by: str, n: int) -> list[dict]:
    r = requests.get(INAT, params={
        "taxon_id": taxon_id, "place_id": 1,
        "quality_grade": "research", "captive": "false",
        "photo_license": "cc0,cc-by", "order_by": order_by,
        "per_page": n,
    }, timeout=30)
    r.raise_for_status()
    out = []
    for obs in r.json()["results"]:
        for p in obs.get("photos", [])[:1]:
            url = p["url"]
            out.append({
                "photo_id": p["id"],
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
        # votes-ordered finds "popular" shots; recency adds variety since
        # popularity correlates poorly with reference quality
        seen, cands = set(), []
        for order in ("votes", "created_at"):
            for c in fetch(sp["taxon_id"], order, args.per_species):
                if c["photo_id"] not in seen:
                    seen.add(c["photo_id"])
                    cands.append(c)
            time.sleep(1.0)
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
