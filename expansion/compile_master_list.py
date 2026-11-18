"""Compile the formal post-expansion Master Species List for AnimlVision.

Implements the scoping rules agreed upon in Lore session b75840e5:
  1. Default eligibility bar: >= 100 US research-grade wild observations
     across Aves, Mammalia, Reptilia, Amphibia (1,671 species).
  2. Retain all Current-100 species (including <100 US obs 'Legendaries':
     Jaguar, Ocelot, Nile Monitor, Walrus, Black-footed Ferret) + all 25 v4
     style anchors + all 20 expansion test-batch species.
  3. Curated 50-99 US obs exceptions: charismatic/endangered US natives and
     established exotic invasives.
  4. Iconic Sharks & Rays (Elasmobranchii): >= 100 US obs + iconic pelagic
     sharks (Great Hammerhead, Blue Shark, Basking Shark, Shortfin Mako,
     Smalltooth Sawfish, Whale Shark, Caribbean Reef Shark).
  5. Tag cryptic/photo-unidentifiable complexes (cryptic salamanders, pocket
     mice/shrews/voles, Empidonax flycatchers, Myotis bats) with
     `id_mode: "genus_candidates"` vs `"direct_species"` so the CV pipeline
     knows when to use the genus-level candidate picker.

Outputs:
  - expansion/master_species_list.json
  - expansion/MASTER_SPECIES_LIST.md
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COUNTS_PATH = REPO / "expansion" / "us_species_counts.json"
CUR100_PATH = REPO / "expansion" / "current_100_us_counts.json"
BATCH20_PATH = REPO / "expansion" / "test_batch_20.json"
OUT_JSON = REPO / "expansion" / "master_species_list.json"
OUT_MD = REPO / "expansion" / "MASTER_SPECIES_LIST.md"

V4_ANCHOR_SLUGS = {
    "american_marten", "elk", "caribou", "white-tailed_deer", "axis_deer",
    "mountain_goat", "nine-banded_armadillo", "harbor_seal", "rhesus_macaque",
    "snowy_owl", "desert_tortoise", "american_robin", "downy_woodpecker",
    "wood_duck", "great_egret", "barking_treefrog", "spring_peeper",
    "american_toad", "eastern_red-backed_salamander", "common_box_turtle",
    "gopher_snake", "western_fence_lizard", "eastern_fence_lizard",
}

# Charismatic US natives & established invasives from the 50-99 US obs band
CHARISMA_AND_INVASIVE_50_99 = {
    # Mammals (50-99)
    "Dipodomys ingens",          # Giant Kangaroo Rat (95)
    "Dipodomys spectabilis",     # Banner-tailed Kangaroo Rat (78)
    "Myotis grisescens",         # Gray Bat (88)
    "Macrotus californicus",     # California Leaf-nosed Bat (85)
    "Neotamias alpinus",         # Alpine Chipmunk (88)
    "Globicephala macrorhynchus",# Short-finned Pilot Whale (68)
    "Sylvilagus idahoensis",     # Pygmy Rabbit (65)
    # Birds (50-99)
    "Pheucticus chrysopeplus",   # Yellow Grosbeak (98)
    "Brotogeris versicolurus",   # White-winged Parakeet (93, invasive)
    "Coereba flaveola",          # Bananaquit (89)
    "Hydrobates homochroa",      # Ashy Storm-Petrel (89)
    "Francolinus francolinus",   # Black Francolin (88, invasive)
    "Phaethon aethereus",        # Red-billed Tropicbird (85)
    "Euptilotis neoxenus",       # Eared Quetzal (82)
    "Pterodroma sandwichensis",  # Hawaiian Petrel (80)
    "Gracula religiosa",         # Common Hill Myna (80, invasive)
    "Chrysolophus pictus",       # Golden Pheasant (78, invasive)
    "Spindalis zena",            # Western Spindalis (75)
    "Corvus imparatus",          # Tamaulipas Crow (73)
    "Magumma parva",             # 'Anianiau (73, Hawaiian endemic)
    # Reptiles (50-99)
    "Pelodiscus sinensis",       # Chinese Softshell Turtle (99, invasive)
    "Leptodeira septentrionalis",# Northern Cat-eyed Snake (91)
    "Uma thurmanae",             # Mohawk Dunes Fringe-toed Lizard (87)
    "Aspidoscelis motaguae",     # Giant Whiptail (84, invasive)
    "Coleonyx reticulatus",      # Reticulated Gecko (80)
    "Anolis allisoni",           # Allison's Anole (75, invasive)
    "Graptemys flavimaculata",   # Yellow-blotched Map Turtle (75)
    "Thamnophis gigas",          # Giant Garter Snake (74)
    "Graptemys gibbonsi",        # Pascagoula Map Turtle (74)
    "Trimorphodon vilkinsonii",  # Texas Lyre Snake (72)
    "Charina umbratica",         # Southern Rubber Boa (70)
    "Pseudemys alabamensis",     # Alabama Red-bellied Cooter (62)
    "Thamnophis rufipunctatus",  # Narrowhead Garter Snake (61)
    "Anolis cybotes",            # Hispaniolan Stout Anole (57, invasive)
    "Lichanura trivirgata",      # Desert Rosy Boa (57)
    "Oxybelis microphthalmus",   # Thornscrub Vine Snake (57)
    # Amphibians (50-99)
    "Necturus beyeri",           # Gulf Coast Waterdog (99)
    "Dicamptodon aterrimus",     # Idaho Giant Salamander (98)
    "Rana muscosa",              # Southern Mountain Yellow-legged Frog (93)
    "Pseudotriton flavissimus",  # Gulf Coast Mud Salamander (88)
    "Anaxyrus retiformis",       # Sonoran Green Toad (70)
    "Aneides caryaensis",        # Hickory Nut Gorge Green Salamander (65)
    "Plethodon idahoensis",      # Coeur d'Alene Salamander (60)
}

# Iconic pelagic/coastal sharks & rays below 100 obs
ICONIC_SHARKS_UNDER_100 = {
    "Sphyrna mokarran",          # Great Hammerhead (78)
    "Prionace glauca",           # Blue Shark (84)
    "Cetorhinus maximus",        # Basking Shark (77)
    "Pristis pectinata",         # Smalltooth Sawfish (85)
    "Isurus oxyrinchus",         # Shortfin Mako (53)
    "Carcharhinus perezii",      # Caribbean Reef Shark (62)
    "Rhincodon typus",           # Whale Shark (48)
    "Galeocerdo cuvier",         # Tiger Shark (115)
}

# Genera where single-photo species ID is notoriously cryptic (flag for genus-level candidate UX)
CRYPTIC_GENERA = {
    "Peromyscus", "Chaetodipus", "Perognathus", "Reithrodontomys", "Microtus",
    "Sorex", "Cryptotis", "Myotis", "Empidonax", "Batrachoseps", "Desmognathus",
}


def rarity_tier(obs: int) -> str:
    if obs >= 10000:
        return "Common"
    if obs >= 1000:
        return "Uncommon"
    if obs >= 100:
        return "Rare"
    return "Legendary"


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    raw = json.loads(COUNTS_PATH.read_text(encoding="utf-8"))
    cur100 = json.loads(CUR100_PATH.read_text(encoding="utf-8"))
    batch20 = json.loads(BATCH20_PATH.read_text(encoding="utf-8"))

    cur100_by_tid = {x["taxon_id"]: x for x in cur100}
    cur100_by_sci = {x["scientific_name"].lower(): x for x in cur100}
    # Map species-level Ursus arctos to Grizzly Bear in current_100 so they don't duplicate
    for x in cur100:
        parts = x["scientific_name"].lower().split()
        if len(parts) == 3:
            cur100_by_sci[f"{parts[0]} {parts[1]}"] = x
    batch20_tids = {x["taxon_id"] for x in batch20}

    master: list[dict] = []
    seen_tids: set[int] = set()

    class_map = {
        "Aves": "Aves",
        "Mammalia": "Mammalia",
        "Reptilia": "Reptilia",
        "Amphibia": "Amphibia",
        "Elasmobranchii (sharks/rays, optional)": "Elasmobranchii",
    }

    for raw_cls, items in raw.items():
        cls_name = class_map[raw_cls]
        for sp in items:
            tid = sp["taxon_id"]
            sci = sp["name"]
            common = sp.get("common_name") or sci
            obs = sp["observations"]
            slug = slugify(common)

            in_cur100 = (tid in cur100_by_tid) or (sci.lower() in cur100_by_sci)
            in_batch20 = tid in batch20_tids
            in_v4_anchors = slug in V4_ANCHOR_SLUGS

            include_reason = None
            if in_cur100:
                include_reason = "current_100"
            elif in_v4_anchors:
                include_reason = "v4_style_anchor"
            elif in_batch20:
                include_reason = "test_batch_20"
            elif cls_name != "Elasmobranchii" and obs >= 100:
                include_reason = "us_obs_gte_100"
            elif cls_name != "Elasmobranchii" and sci in CHARISMA_AND_INVASIVE_50_99:
                include_reason = "curated_50_99_exception"
            elif cls_name == "Elasmobranchii" and (obs >= 100 or sci in ICONIC_SHARKS_UNDER_100):
                include_reason = "iconic_elasmobranch"

            if not include_reason:
                continue

            genus = sci.split(" ")[0]
            # Well-known species in cryptic genera (e.g., Eastern Red-backed Salamander, Gray Bat) stay direct
            is_cryptic = (genus in CRYPTIC_GENERA) and not (in_cur100 or in_v4_anchors or in_batch20 or obs >= 15000)

            cat_info = cur100_by_tid.get(tid) or cur100_by_sci.get(sci.lower())
            origin = cat_info["category"] if cat_info else "Native"
            if cat_info:
                common = cat_info["name"]
                slug = slugify(common)
                seen_tids.add(cat_info["taxon_id"])

            master.append({
                "taxon_id": tid,
                "common_name": common,
                "scientific_name": sci,
                "slug": slug,
                "class": cls_name,
                "us_research_grade_obs": obs,
                "rarity": rarity_tier(obs),
                "origin": origin,
                "include_reason": include_reason,
                "in_current_100": in_cur100,
                "in_v4_anchors": in_v4_anchors,
                "in_test_batch_20": in_batch20,
                "id_mode": "genus_candidates" if is_cryptic else "direct_species",
            })
            seen_tids.add(tid)

    # Ensure any current_100 species missing from raw US counts (e.g., if taxon_id differed) are present
    for c in cur100:
        if c["taxon_id"] not in seen_tids:
            slug = slugify(c["name"])
            master.append({
                "taxon_id": c["taxon_id"],
                "common_name": c["name"],
                "scientific_name": c["scientific_name"],
                "slug": slug,
                "class": "Unknown",
                "us_research_grade_obs": c["us_research_grade_obs"],
                "rarity": rarity_tier(c["us_research_grade_obs"]),
                "origin": c["category"],
                "include_reason": "current_100",
                "in_current_100": True,
                "in_v4_anchors": slug in V4_ANCHOR_SLUGS,
                "in_test_batch_20": False,
                "id_mode": "direct_species",
            })

    master.sort(key=lambda x: (-x["us_research_grade_obs"], x["common_name"]))

    by_class = {}
    by_reason = {}
    by_rarity = {}
    by_id_mode = {}
    for m in master:
        by_class[m["class"]] = by_class.get(m["class"], 0) + 1
        by_reason[m["include_reason"]] = by_reason.get(m["include_reason"], 0) + 1
        by_rarity[m["rarity"]] = by_rarity.get(m["rarity"], 0) + 1
        by_id_mode[m["id_mode"]] = by_id_mode.get(m["id_mode"], 0) + 1

    payload = {
        "summary": {
            "total_species": len(master),
            "by_class": by_class,
            "by_include_reason": by_reason,
            "by_rarity": by_rarity,
            "by_id_mode": by_id_mode,
        },
        "species": master,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write clean Markdown summary
    md_lines = [
        "# AnimlVision Post-Expansion Master Species List",
        "",
        "Compiled from [`expansion/us_species_counts.json`](us_species_counts.json) and [`expansion/current_100_us_counts.json`](current_100_us_counts.json) per the scoping rules in Lore session `b75840e5`.",
        "",
        "## 1. Summary Totals",
        "",
        f"- **Total Master Roster:** **{len(master):,} species**",
        f"  - **Direct Single-Species ID (`direct_species`):** **{by_id_mode.get('direct_species', 0):,} species**",
        f"  - **Cryptic Genus-Candidate Picker (`genus_candidates`):** **{by_id_mode.get('genus_candidates', 0):,} species** *(pocket mice, shrews, voles, Myotis bats, Empidonax flycatchers, Desmognathus/Batrachoseps salamanders)*",
        "",
        "### Breakdown by Vertebrate Class",
        "| Class | Total Species |",
        "|---|---|",
    ]
    for k, v in by_class.items():
        md_lines.append(f"| **{k}** | {v:,} |")

    md_lines += [
        "",
        "### Breakdown by Inclusion Rule",
        "| Inclusion Rule | Count | Description |",
        "|---|---|---|",
        f"| `current_100` | {by_reason.get('current_100', 0):,} | Original 100 Dex species (including `<100` US obs Legendaries: Jaguar, Ocelot, Nile Monitor, Walrus, Black-footed Ferret) |",
        f"| `test_batch_20` / `v4_style_anchor` | {by_reason.get('test_batch_20', 0) + by_reason.get('v4_style_anchor', 0):,} | Expansion test batch & newly locked v4 style anchors |",
        f"| `us_obs_gte_100` | {by_reason.get('us_obs_gte_100', 0):,} | Additional US non-fish tetrapods clearing `>= 100` research-grade wild observations |",
        f"| `curated_50_99_exception` | {by_reason.get('curated_50_99_exception', 0):,} | Hand-picked charismatic US natives & established invasives in the `50–99` US obs band |",
        f"| `iconic_elasmobranch` | {by_reason.get('iconic_elasmobranch', 0):,} | Iconic US coastal & pelagic Sharks, Rays, and Skates |",
        "",
        "### Breakdown by Rarity Tier (from US Wild Research-Grade Observations)",
        "| Rarity Tier | Observation Range | Species Count |",
        "|---|---|---|",
        f"| **Common** | `>= 10,000` US obs | {by_rarity.get('Common', 0):,} |",
        f"| **Uncommon** | `1,000 – 9,999` US obs | {by_rarity.get('Uncommon', 0):,} |",
        f"| **Rare** | `100 – 999` US obs | {by_rarity.get('Rare', 0):,} |",
        f"| **Legendary** | `< 100` US obs (curated charisma/legendaries) | {by_rarity.get('Legendary', 0):,} |",
        "",
        "## 2. All `< 100` US Obs 'Legendary' & `50–99` Exception Species Included",
        "| Common Name | Scientific Name | Class | US Obs | Reason |",
        "|---|---|---|---|---|",
    ]
    for m in master:
        if m["us_research_grade_obs"] < 100:
            md_lines.append(
                f"| {m['common_name']} | *{m['scientific_name']}* | {m['class']} | {m['us_research_grade_obs']} | `{m['include_reason']}` |"
            )

    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
