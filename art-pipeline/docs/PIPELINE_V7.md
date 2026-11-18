# Pipeline v7 Architecture & Technical Notes

`art-pipeline/animl_art/` encapsulates the production **v7** wildlife illustration pipeline built around `animl_style_v7-step00002000.safetensors` and the **`v3` smooth curvature-tapered outline equalizer**.

## VRAM-Isolated Pipeline Stages (`16 GB RTX 5080`)

Calling `python animl.py art run --batch <name> --phase all` executes three sequential subprocesses so GPU memory never overlaps across large models:

1. **Stage 1A — Photo Fetch & Cutout (`animl_art/vetter.py`, `sources.py`, `matte.py`)**
   - Terrestrial species pull from iNaturalist Curated Taxon Photos, Wikimedia Commons, and iNaturalist research-grade observations.
   - Marine species (`subtype == "shark_ray"`) use `fetch_marine_candidates_v7` to interleave underwater scuba observations & filtered Commons media, ranked via **Dual-CLIP** (`0.55 * scuba_water_score + 0.45 * cutout_anatomy_score`).
   - Segmentation uses 1024px BiRefNet ONNX (`matte.remove`) with **connected hysteresis** (`binary_propagation` down to `alpha >= 38` for marine whip tails / `90` otherwise) and automatic underwater cyan/green LAB neutralization.
2. **Stage 1B — Wikipedia Clauses & Gemma3:27b Vetting (`vetter.py`, `sources.py`)**
   - `fetch_wiki_clauses_map` extracts visual morphology clauses from Wikipedia Description/lead sections (falling back to scientific binomial on disambiguations).
   - `vet_side_by_side_and_extract_traits_v7` passes `[Original Photo | Cutout]` side-by-side to `gemma3:27b` (`180s` timeout), rejecting above-water surface-ripple shots, dead/held specimens, fused rocks/leaves, or aquarium glass glare, and enforcing diagnostic field marks, `unstriped` mammal coats, furred paws/muzzles, and dorsal-vs-ventral scoping on rays.
   - Ollama is explicitly unloaded (`keep_alive=0`) when Stage 1B exits.
3. **Stage 3 — Depth + Canny + SDXL MultiControlNet + Post-Clean & v3 Equalizer (`stylize.py`, `postprocess.py`)**
   - **Depth-Anything-V2-Small** computes foreground-normalized depth maps (`40..255`), then unloads from VRAM.
   - **Pre-stylization conditioning (`prepare_conditioned_cutout_and_canny_v7`)**:
     - *Mammal pale-collar conditioner*: uses foreground-masked LAB `MORPH_CLOSE` so grizzled cream collars/shoulder stripes do not average to mid-brown.
     - *Countershaded Elasmobranch ventral conditioner*: uses `MORPH_OPEN` (`19x19` seed + `9x9` propagation) to isolate smooth shaded white ventral wings/bellies (`L ~ 97–132`), lifts them cleanly to matte gouache cream-white (`L = 202..218`) without touching dorsal white spots or crossing mouth/snout creases, and clears faint interior throat-shadow Canny edges.
   - **Prompt & Strength Calibration (`build_prompt_and_neg_v7`)**:
     - `sanitize_species_subject_v7` rewrites misleading common names (e.g., *Prairie Dog -> Cynomys prairie marmot*, *Sea Lion -> eared seal pinniped*, *Alligator Lizard -> Elgaria lizard*).
     - Bird wing-red bleed guard prevents red plumage on wings from leaking onto heads/eyebrows unless red actually occurs on the head.
     - Batoidea rays (`is_batoidea_ray_v7`) automatically route through `animl_style_v7_unet_only.safetensors` (`strength=0.44`, `canny=0.54`, full `1.0` LoRA scale) to prevent text-encoder `leopard_shark` / `reef_fish` bleed.
   - **Post-Clean & v3 Equalizer (`postprocess.py`)**:
     - `clean_post_v7` runs BiRefNet BEFORE bounding-box cropping & white-background flattening, restoring dark perches and thin marine whip tails via connected hysteresis.
     - `equalize_outline_smooth_tapered` applies fur-fitted silhouette trimming, `4x` supersampled Euclidean distance tapering, fur-tinted outer charcoal-umber ink, and Hessian-anisotropic internal structural crease darkening.

## Training Dataset (`art-pipeline/lora/dataset_v7/`)
- Built deterministically on CPU by `art-pipeline/lora/build_dataset_v7.py` from 45 selected `v3`-equalized candidates in `art-pipeline/equalized_candidates_78/`.
- `dataset_v7.toml` + `train_v7.ps1` train SDXL Rank-32 LoRA (`2,000` steps), with `animl_style_v7-step00002000.safetensors` and `animl_style_v7_unet_only.safetensors` as the production weights.
