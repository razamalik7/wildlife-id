# Maintenance Scripts

Standalone operational utilities moved out of the repository root.

- `parks/` — Scripts and validated JSON indices for expanding and auditing National Park `place_id` coverage (`manage_parks.py`, `validate_nps.py`, `analyze_parks.py`, `expand_parks_*.py`).
- `taxonomy/` — Taxonomy cross-check utilities (`audit_taxonomy.py`).

Run via the top-level CLI:
```bash
python animl.py parks validate
python animl.py parks manage --help
```
