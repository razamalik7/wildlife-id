"""Scope the Dex expansion with real iNaturalist data.

For each non-fish vertebrate class (plus sharks/rays as an optional line),
pull US research-grade wild species counts and report how many species clear
various minimum-observation thresholds. Observation count is our proxy for
"enough training photos exist" — the real wall on expansion.

Writes expansion/us_species_counts.json (full per-species counts) and prints
the threshold table.

Usage: python expansion/species_scoping.py
"""

import json
import time
from pathlib import Path

import requests

API = "https://api.inaturalist.org/v1/observations/species_counts"
PLACE_US = 1  # iNaturalist place_id for the United States

CLASSES = {
    "Aves": 3,
    "Mammalia": 40151,
    "Reptilia": 26036,
    "Amphibia": 20978,
    "Elasmobranchii (sharks/rays, optional)": 505362,
}

THRESHOLDS = [100, 500, 1000, 5000, 10000]


def fetch_class(taxon_id: int) -> list[dict]:
    out, page = [], 1
    while True:
        r = requests.get(API, params={
            "place_id": PLACE_US,
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "captive": "false",
            "per_page": 500,
            "page": page,
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data["results"]
        for row in results:
            t = row["taxon"]
            if t.get("rank") != "species":
                continue
            out.append({
                "taxon_id": t["id"],
                "name": t["name"],
                "common_name": t.get("preferred_common_name"),
                "observations": row["count"],
            })
        if page * 500 >= data["total_results"] or not results:
            return out
        page += 1
        time.sleep(1.1)  # stay friendly to the API


def main() -> None:
    all_data = {}
    for cls, tid in CLASSES.items():
        species = fetch_class(tid)
        all_data[cls] = species
        print(f"{cls}: {len(species)} species with >=1 research-grade US observation")
        time.sleep(1.1)

    Path("expansion/us_species_counts.json").write_text(json.dumps(all_data, indent=1))

    header = "threshold | " + " | ".join(c.split(" ")[0] for c in CLASSES) + " | total (no sharks) | total (with)"
    print("\n" + header)
    print("-" * len(header))
    for th in THRESHOLDS:
        counts = {c: sum(1 for s in sp if s["observations"] >= th) for c, sp in all_data.items()}
        no_sharks = sum(v for c, v in counts.items() if not c.startswith("Elasmobranchii"))
        with_sharks = sum(counts.values())
        print(f">= {th:>6,} | " + " | ".join(str(counts[c]) for c in CLASSES)
              + f" | {no_sharks} | {with_sharks}")


if __name__ == "__main__":
    main()
