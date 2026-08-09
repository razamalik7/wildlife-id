"""Assemble the first-20 expansion test batch.

Selection strategy (see expansion/TEST_BATCH.md for full rationale):
- one 5-species duck cluster (tests batch-by-cluster art + dense CV neighborhood)
- deliberate confusables with the existing 100 (mule deer, muskrat, pond
  slider, gopher snake, gopher tortoise, American toad)
- highest-observed gaps (robin, fence lizard) and new art body plans
  (clinging woodpecker, pika, treefrog, salamander)

For each species: verify taxon_id against iNat, confirm it is not already in
species_config.json, record observation count, and fetch up to 3 CC0/CC-BY
research-grade reference-photo candidates for the art pipeline.

Writes expansion/test_batch_20.json and appends candidates to
art-pipeline/references.csv.

Usage: python expansion/build_test_batch.py
"""

import csv
import json
import time
from pathlib import Path

import requests

# (common name as we'd display it, iNat scientific name, cluster/rationale tag)
PICKS = [
    # Duck cluster - batch-by-cluster test, joins existing Mallard
    ("Wood Duck", "Aix sponsa", "duck-cluster"),
    ("Northern Shoveler", "Spatula clypeata", "duck-cluster"),
    ("Northern Pintail", "Anas acuta", "duck-cluster"),
    ("American Wigeon", "Mareca americana", "duck-cluster"),
    ("Green-winged Teal", "Anas crecca", "duck-cluster"),
    # Iconic gaps - cheap adds, huge data
    ("American Robin", "Turdus migratorius", "most-observed US bird"),
    ("Great Egret", "Ardea alba", "iconic; white-plumage rendering test"),
    ("Downy Woodpecker", "Dryobates pubescens", "new body plan: clinging pose"),
    # Mammals
    ("Mule Deer", "Odocoileus hemionus", "confusable: white-tailed deer/elk"),
    ("Muskrat", "Ondatra zibethicus", "completes beaver/nutria/muskrat trio"),
    ("California Ground Squirrel", "Otospermophilus beecheyi", "joins squirrel cluster"),
    ("American Pika", "Ochotona princeps", "iconic small mammal, distinctive"),
    # Reptiles
    ("Western Fence Lizard", "Sceloporus occidentalis", "most-observed US reptile"),
    ("Pond Slider", "Trachemys scripta", "confusable: painted turtle"),
    ("Gopher Snake", "Pituophis catenifer", "confusable: rattlesnake mimic"),
    ("Common Watersnake", "Nerodia sipedon", "high-obs; snake pose variety"),
    ("Gopher Tortoise", "Gopherus polyphemus", "geo-prior test vs desert tortoise (disjoint ranges)"),
    # Amphibians
    ("American Toad", "Anaxyrus americanus", "confusable: cane toad (native vs invasive)"),
    ("Spring Peeper", "Pseudacris crucifer", "tiny treefrog body plan"),
    ("Eastern Red-backed Salamander", "Plethodon cinereus", "most-observed salamander; elongate body plan"),
]

INAT = "https://api.inaturalist.org/v1"
PLACE_US = 1


def resolve(sci_name: str) -> dict:
    r = requests.get(f"{INAT}/taxa", params={"q": sci_name, "rank": "species"}, timeout=30)
    r.raise_for_status()
    for t in r.json()["results"]:
        if t["name"].lower() == sci_name.lower():
            return t
    raise ValueError(f"taxon not found: {sci_name}")


def us_observation_count(taxon_id: int) -> int:
    r = requests.get(f"{INAT}/observations", params={
        "taxon_id": taxon_id, "place_id": PLACE_US,
        "quality_grade": "research", "captive": "false", "per_page": 0,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["total_results"]


def photo_candidates(taxon_id: int, n: int = 3) -> list[dict]:
    r = requests.get(f"{INAT}/observations", params={
        "taxon_id": taxon_id, "place_id": PLACE_US,
        "quality_grade": "research", "captive": "false",
        "photo_license": "cc0,cc-by", "order_by": "votes",
        "per_page": n,
    }, timeout=30)
    r.raise_for_status()
    out = []
    for obs in r.json()["results"]:
        for p in obs.get("photos", [])[:1]:
            out.append({
                "url": p["url"].replace("square", "original"),
                "license": p.get("license_code"),
                "attribution": p.get("attribution"),
                "observation": f"https://www.inaturalist.org/observations/{obs['id']}",
            })
    return out


def main() -> None:
    current_ids = {
        s.get("taxonomy", {}).get("taxon_id")
        for s in json.loads(Path("frontend/public/species_config.json").read_text(encoding="utf-8"))
    }

    batch = []
    for common, sci, tag in PICKS:
        taxon = resolve(sci)
        time.sleep(1.0)
        if taxon["id"] in current_ids:
            print(f"!! {common} already in species_config (taxon {taxon['id']}) - replace this pick")
            continue
        count = us_observation_count(taxon["id"])
        time.sleep(1.0)
        photos = photo_candidates(taxon["id"])
        time.sleep(1.0)
        batch.append({
            "common_name": common,
            "scientific_name": taxon["name"],
            "taxon_id": taxon["id"],
            "us_research_grade_obs": count,
            "rationale": tag,
            "reference_photo_candidates": photos,
        })
        print(f"ok {common}: taxon {taxon['id']}, {count:,} obs, {len(photos)} photo candidates")

    Path("expansion/test_batch_20.json").write_text(json.dumps(batch, indent=1))

    refs = Path("art-pipeline/references.csv")
    with refs.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for sp in batch:
            slug = sp["common_name"].lower().replace(" ", "_")
            for i, p in enumerate(sp["reference_photo_candidates"]):
                w.writerow([slug, p["url"], p["license"], p["attribution"],
                            f"candidate {i+1} (auto; verify pose/quality); {p['observation']}"])
    print(f"\nwrote expansion/test_batch_20.json ({len(batch)} species) and appended candidates to references.csv")


if __name__ == "__main__":
    main()
