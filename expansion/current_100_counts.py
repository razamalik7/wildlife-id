"""US observation counts for the CURRENT 100 species.

Same methodology as species_scoping.py: place_id=1 (United States,
admin_level 0 — includes AK/HI/territorial waters), research-grade,
wild only (captive=false). Queries each config taxon_id directly so
counts are exact per species, not name-matched.

Writes expansion/current_100_us_counts.json, prints ascending.

Usage: python expansion/current_100_counts.py
"""

import json
import time
from pathlib import Path

import requests

INAT = "https://api.inaturalist.org/v1/observations"


def us_count(taxon_id: int) -> int:
    r = requests.get(INAT, params={
        "taxon_id": taxon_id, "place_id": 1,
        "quality_grade": "research", "captive": "false", "per_page": 0,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["total_results"]


def main() -> None:
    config = json.loads(Path("frontend/public/species_config.json").read_text(encoding="utf-8"))
    rows = []
    for s in config:
        tid = s.get("taxonomy", {}).get("taxon_id")
        name = s["name"]
        if not tid:
            print(f"!! {name}: no taxon_id in config")
            continue
        count = us_count(tid)
        rows.append({
            "name": name,
            "scientific_name": s.get("scientific_name"),
            "taxon_id": tid,
            "category": s.get("category"),
            "us_research_grade_obs": count,
        })
        time.sleep(1.0)

    rows.sort(key=lambda r: r["us_research_grade_obs"])
    Path("expansion/current_100_us_counts.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['us_research_grade_obs']:>8,}  {r['name']} ({r['category']})")


if __name__ == "__main__":
    main()
