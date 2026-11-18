# AnimlVision 1,777-Species Expansion Catalog

Defines the expanded North American fauna roster beyond the initial 100 Anidex species.

## Key Files
- `master_species_list.json` — Master list of **1,777 species** (`826 Aves`, `282 Mammalia`, `360 Reptilia`, `251 Amphibia`, `58 Elasmobranchii`), split into `1,702` direct single-species IDs and `75` cryptic `genus_candidates`.
- `MASTER_SPECIES_LIST.md` — Summary statistics, rarity tiers, and `<100` US observation exceptions.
- `compile_master_list.py` — Recompiles `master_species_list.json` from `us_species_counts.json`, `current_100_us_counts.json`, and `species_scoping.py`.
- `TEST_BATCH.md` / `test_batch_20.json` — 20-species pilot batch specification used to validate the expansion workflow.

Run from repo root:
```bash
python animl.py expansion compile
```
