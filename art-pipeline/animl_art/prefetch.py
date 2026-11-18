"""CPU-only Top-500 manifest builder and batched iNaturalist / Wikipedia pre-warmer."""

import json
import re
import time
from pathlib import Path

import requests

from animl_art import (
    BATCHES_DIR,
    BENCHMARKS_DIR,
    CACHE_DIR,
    EXPANSION_LIST_PATH,
    LORA_DIR,
    MASTERS_DIR,
    TIERS_DIR,
)
from animl_art.sources import (
    HEADERS,
    extract_visual_clauses,
    fetch_wiki_batch,
    get_inat_taxa_cache,
    get_wiki_clauses_cache,
)

# Balanced class quotas summing to 500 across all 5 vertebrate classes (~10x50 tiers)
BALANCED_QUOTAS = {
    "Aves": 240,
    "Mammalia": 90,
    "Reptilia": 90,
    "Amphibia": 55,
    "Elasmobranchii": 25,
}

_SNAKE_KEYWORDS = (
    "snake", "boa", "python", "adder", "viper", "cobra", "racer", "coachwhip",
    "cottonmouth", "copperhead", "rattlesnake", "garter", "watersnake", "milksnake",
    "ratsnake", "kingsnake", "massasauga", "moccasin", "wormsnake",
)
_TURTLE_KEYWORDS = ("turtle", "tortoise", "terrapin", "cooter", "slider", "softshell", "map turtle")
_GROUND_BIRD_KEYWORDS = (
    "duck", "goose", "swan", "heron", "egret", "plover", "killdeer", "sandpiper",
    "gull", "tern", "grebe", "quail", "pheasant", "turkey", "grouse", "crane",
    "rail", "pelican", "cormorant", "vulture", "mallard", "teal", "merganser",
    "coot", "moorhen", "ibis", "spoonbill", "stilt", "avocet", "curlew", "snipe",
)
_AQUATIC_MAMMAL_KEYWORDS = ("whale", "dolphin", "porpoise", "manatee", "orca", "beluga", "narwhal")
_LEGLESS_REPTILE_KEYWORDS = ("glass lizard", "legless lizard", "worm lizard")


def infer_subtype(common: str, cls_name: str) -> tuple[str, bool]:
    """Infer `(subtype, default_perched)` from taxonomic class and common name."""
    c_low = common.lower()
    if cls_name == "Amphibia":
        return "amphibian", False
    if cls_name == "Elasmobranchii":
        return "shark_ray", False
    if cls_name == "Reptilia":
        if any(k in c_low for k in _LEGLESS_REPTILE_KEYWORDS) or any(k in c_low for k in _SNAKE_KEYWORDS):
            return "snake", False
        if any(k in c_low for k in _TURTLE_KEYWORDS):
            return "turtle", False
        return "lizard", False
    if cls_name == "Aves":
        if any(k in c_low for k in _GROUND_BIRD_KEYWORDS):
            return "bird_ground", False
        return "bird_perched", True
    if cls_name == "Mammalia":
        if any(k in c_low for k in _AQUATIC_MAMMAL_KEYWORDS):
            return "mammal_aquatic", False
        if re.search(r"\bbat\b", c_low) or any(k in c_low for k in ("myotis", "pipistrelle")):
            return "mammal_bat", False
    return "mammal_ground", False


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _training_and_done_sets() -> tuple[set[str], set[str], set[int]]:
    """Collect LoRA training stems (v6+v7), existing masters, and completed v7_unseen_10 species."""
    train_stems: set[str] = set()
    for folder in ("dataset", "dataset_v7"):
        d = LORA_DIR / folder
        if d.exists():
            for p in d.rglob("*.txt"):
                s = re.sub(r"^cand_\d+_", "", p.stem).lower()
                train_stems.add(s)
                train_stems.add(re.sub(r"_\d+$", "", s))

    masters_stems = {p.stem.lower() for p in MASTERS_DIR.glob("*.png") if not p.name.startswith("icon-")} if MASTERS_DIR.exists() else set()
    completed_tids: set[int] = set()
    completed_names: set[str] = set()
    u10_meta = BENCHMARKS_DIR / "unseen_10_batch1_metadata.json"
    if u10_meta.exists():
        for v in json.loads(u10_meta.read_text(encoding="utf-8")).values():
            completed_tids.add(int(v["taxon_id"]))
            completed_names.add(v["common_name"].lower())
    return train_stems, masters_stems | completed_names, completed_tids


def _format_manifest_entries(rows: list[dict]) -> list[dict]:
    out = []
    for idx, e in enumerate(rows, start=1):
        com, sci, tid, cls_n = e["common_name"], e["scientific_name"], int(e["taxon_id"]), e["class"]
        sub, d_perch = infer_subtype(com, cls_n)
        out.append(
            {
                "stem": f"{idx:03d}_{_slugify(com)}",
                "common_name": com,
                "scientific_name": sci,
                "taxon_id": tid,
                "class": cls_n,
                "subtype": sub,
                "default_perched": d_perch,
                "us_research_grade_obs": int(e.get("us_research_grade_obs", 0)),
            }
        )
    return out


def build_top500_manifests() -> dict:
    """Build `top500_inat_balanced.json` (plus 10x50 tiers) and `top500_inat_raw.json` strictly on CPU."""
    train_stems, done_names, done_tids = _training_and_done_sets()
    catalog = json.loads(EXPANSION_LIST_PATH.read_text(encoding="utf-8"))["species"]

    eligible: list[dict] = []
    for e in catalog:
        if e.get("id_mode") != "direct_species":
            continue
        if e.get("in_current_100", False):
            continue
        slug = e["slug"].lower()
        slug_u = slug.replace("-", "_")
        com_l = e["common_name"].lower()
        com_u = _slugify(com_l)
        if slug in train_stems or slug_u in train_stems or com_u in train_stems:
            continue
        if slug in done_names or slug_u in done_names or com_u in done_names or com_l in done_names:
            continue
        if int(e["taxon_id"]) in done_tids:
            continue
        eligible.append(e)

    eligible.sort(key=lambda x: (-int(x.get("us_research_grade_obs", 0)), int(x["taxon_id"])))
    raw_500 = eligible[:500]

    by_cls: dict[str, list[dict]] = {}
    for e in eligible:
        by_cls.setdefault(e["class"], []).append(e)

    # Interleave the Top 500 balanced pool into 10 progressive 50-animal tiers preserving quota ratios
    tier_quotas = {cls_n: quota // 10 for cls_n, quota in BALANCED_QUOTAS.items()}
    # Sums: 24 + 9 + 9 + 5.5 -> alternating 5/6 Amphibia & 2/3 Elasmobranchii across tiers
    balanced_500_rows: list[dict] = []
    cursors = {c: 0 for c in BALANCED_QUOTAS}
    TIERS_DIR.mkdir(parents=True, exist_ok=True)
    for t_idx in range(10):
        tier_rows: list[dict] = []
        amph_n = 6 if (t_idx % 2 == 0) else 5
        elasmo_n = 2 if (t_idx % 2 == 0) else 3
        per_tier = {"Aves": 24, "Mammalia": 9, "Reptilia": 9, "Amphibia": amph_n, "Elasmobranchii": elasmo_n}
        for cls_n in ("Mammalia", "Aves", "Reptilia", "Amphibia", "Elasmobranchii"):
            need = per_tier[cls_n]
            chunk = by_cls.get(cls_n, [])[cursors[cls_n] : cursors[cls_n] + need]
            cursors[cls_n] += len(chunk)
            tier_rows.extend(chunk)
        balanced_500_rows.extend(tier_rows)
        (TIERS_DIR / f"tier_{t_idx + 1:02d}.json").write_text(
            json.dumps(_format_manifest_entries(tier_rows), indent=2), encoding="utf-8"
        )

    assert len(balanced_500_rows) == 500, f"Expected 500 balanced rows, got {len(balanced_500_rows)}"
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    bal_path = BATCHES_DIR / "top500_inat_balanced.json"
    raw_path = BATCHES_DIR / "top500_inat_raw.json"
    bal_fmt = _format_manifest_entries(balanced_500_rows)
    raw_fmt = _format_manifest_entries(raw_500)
    bal_path.write_text(json.dumps(bal_fmt, indent=2), encoding="utf-8")
    raw_path.write_text(json.dumps(raw_fmt, indent=2), encoding="utf-8")

    return {
        "eligible_total": len(eligible),
        "balanced_path": str(bal_path),
        "raw_path": str(raw_path),
        "tiers_dir": str(TIERS_DIR),
        "balanced_by_class": dict(BALANCED_QUOTAS),
        "raw_by_class": {c: sum(1 for r in raw_fmt if r["class"] == c) for c in sorted({r["class"] for r in raw_fmt})},
    }


def _load_manifest_items(manifest_paths: list[Path]) -> list[tuple[int, str, str]]:
    """Return deduplicated `[(taxon_id, common_name, scientific_name)]` across manifest files."""
    seen: dict[int, tuple[int, str, str]] = {}
    for mp in manifest_paths:
        data = json.loads(mp.read_text(encoding="utf-8"))
        for row in data:
            if isinstance(row, list):
                com, sci, tid = str(row[1]), str(row[2]), int(row[3])
            else:
                com, sci, tid = str(row["common_name"]), str(row["scientific_name"]), int(row["taxon_id"])
            seen[tid] = (tid, com, sci)
    return list(seen.values())


def prefetch_inat_and_wiki_caches(manifest_paths: list[Path], max_photos_per_taxon: int = 8) -> dict:
    """Batch-warm `inat_taxa_cache.json` and `wiki_clauses_cache.json` cleanly on CPU."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    items = _load_manifest_items(manifest_paths)

    inat_cache_p = CACHE_DIR / "inat_taxa_cache.json"
    wiki_cache_p = CACHE_DIR / "wiki_clauses_cache.json"
    inat_cache = dict(get_inat_taxa_cache())
    wiki_cache = dict(get_wiki_clauses_cache())

    missing_tids = [tid for tid, _, _ in items if str(tid) not in inat_cache]
    print(f"Prefetching iNat taxa photos: {len(missing_tids)} uncached of {len(items)} unique taxa...", flush=True)
    for start in range(0, len(missing_tids), 30):
        chunk = missing_tids[start : start + 30]
        ids_str = ",".join(str(t) for t in chunk)
        for attempt in range(3):
            try:
                r = requests.get(f"https://api.inaturalist.org/v1/taxa/{ids_str}", headers=HEADERS, timeout=30)
                if r.status_code in (429, 502, 503):
                    time.sleep(2.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                for res in r.json().get("results", []):
                    tid_str = str(res["id"])
                    photos = []
                    for tp in res.get("taxon_photos", []):
                        p = tp.get("photo", {})
                        lic = (p.get("license_code") or "").lower()
                        if lic not in ("cc0", "cc-by", "pd", ""):
                            continue
                        u = p.get("url") or p.get("medium_url")
                        if not u:
                            continue
                        photos.append(
                            {
                                "photo_id": f"inat_t_{p['id']}",
                                "source_tier": "inat_curated_taxon",
                                "medium_url": u.replace("square", "medium"),
                                "original_url": u.replace("square", "original"),
                                "license": lic,
                            }
                        )
                        if len(photos) >= max_photos_per_taxon:
                            break
                    inat_cache[tid_str] = {
                        "obs_count": int(res.get("observations_count", 0) or 0),
                        "photos": photos,
                    }
                break
            except Exception as e:
                print(f"  [iNat chunk {start} attempt {attempt+1} warn]: {e}", flush=True)
                time.sleep(1.5)
        time.sleep(0.45)

    for tid in missing_tids:
        inat_cache.setdefault(str(tid), {"obs_count": 0, "photos": []})
    inat_cache_p.write_text(json.dumps(inat_cache, separators=(",", ":")), encoding="utf-8")

    missing_wiki = [(com, sci) for _, com, sci in items if com not in wiki_cache and sci not in wiki_cache]
    print(f"Prefetching Wikipedia clauses: {len(missing_wiki)} uncached of {len(items)} species...", flush=True)
    if missing_wiki:
        raw_com = fetch_wiki_batch([c for c, _ in missing_wiki])
        need_sci: list[tuple[str, str]] = []
        for com, sci in missing_wiki:
            txt = raw_com.get(com, "")
            clauses = extract_visual_clauses(txt) if len(txt) >= 500 and "may refer to:" not in txt[:300].lower() else ""
            if clauses:
                wiki_cache[com] = clauses
            else:
                need_sci.append((com, sci))
        if need_sci:
            raw_sci = fetch_wiki_batch([s for _, s in need_sci])
            for com, sci in need_sci:
                clauses = extract_visual_clauses(raw_sci.get(sci, ""))
                if clauses:
                    wiki_cache[com] = clauses
                    wiki_cache[sci] = clauses
    wiki_cache_p.write_text(json.dumps(wiki_cache, indent=2), encoding="utf-8")

    cov_inat = sum(1 for tid, _, _ in items if inat_cache.get(str(tid), {}).get("photos"))
    cov_wiki = sum(1 for _, com, sci in items if wiki_cache.get(com) or wiki_cache.get(sci))
    return {
        "unique_species": len(items),
        "inat_with_photos": cov_inat,
        "wiki_with_clauses": cov_wiki,
        "inat_cache_kb": round(inat_cache_p.stat().st_size / 1024, 1),
        "wiki_cache_kb": round(wiki_cache_p.stat().st_size / 1024, 1),
    }
