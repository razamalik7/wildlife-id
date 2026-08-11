# Art Factory — Recommended Production Stack

Synthesis of the four research surveys in this directory
([sources.md](sources.md), [graders.md](graders.md),
[generators.md](generators.md), [qa_loops.md](qa_loops.md)) into one
architecture. Design rule: **frontier models (Claude) design and debug the
system; the running system uses local models and sub-cent APIs.**

## Stage 1 — Reference photo sourcing

Priority chain per species:
1. **Wikidata P18** portrait (one SPARQL batch for all 1,400; editor-curated;
   keep only CC0/CC-BY/PD — ~45% pass rate)
2. **iNaturalist open-data S3 bucket** — the workhorse. Monthly license
   metadata TSVs + photos, no auth, no rate limits; whole-corpus candidate
   pull runs offline. `quality_grade=research`, CC0/CC-BY only.
3. **Wikimedia Commons Quality/Valued Images** upgrade pass
4. **Flickr API** (license ids 4/9/10/8) — quality fallback, needs species
   verification by the grader
5. **Smithsonian Open Access / USFWS (PD)** gap-fill; GBIF for taxonomy
   normalization only. Ruled out: Macaulay/eBird, CalPhotos, ADW,
   HerpMapper, Unsplash/Pexels/Pixabay (licenses).

Hygiene: snapshot {license, attribution, url} at fetch (iNat licenses are
mutable); dedupe by photo id + perceptual hash; ship CC-BY credits.

## Stage 2 — Reference audit (input grading)

Deterministic first, VLM second, API tiebreak:
1. **MegaDetector** (already wired, MDV5A): exactly-1 animal, bbox 15–75%,
   no edge contact, sharpness.
2. **DeepLabCut SuperAnimal-Quadruped** keypoints: all-paws-visible check
   (quadrupeds; birds/herps go to the VLM).
3. **Qwen3-VL-8B-Instruct Q4** (replaces Gemma — its multi-image attention
   actually works): pose (flying/feeding/preening = reject), objects in
   mouth/paws, juvenile/atypical, photo-vs-drawing.
4. Escalation: **Gemini Flash-Lite** (~$0.22/1k images, triple-vote cheap),
   **Claude Haiku batch** as disagreement-breaker. Full-corpus audit budget
   at 1,400 species × ~24 candidates: **$30–150 total**.

## Stage 3 — Generation

- **Now (working)**: SDXL + house LoRA (v2/v3) + matte prompt language +
  per-species vignette contexts + cutout img2img @ 0.5–0.6, 4 candidates.
- **Next upgrade (biggest quality lever)**: retrain the style LoRA on
  **Z-Image-Base** (Alibaba, Apache 2.0, 6B, #1 open model; trains on 16GB
  via ai-toolkit) and add **depth+lineart ControlNet from the cutout** —
  anatomy held by structure, so denoise can rise to ~0.8 for full style
  adoption. This directly attacks the style-vs-accuracy tradeoff.
- **Local repair/edit**: **Qwen-Image-Edit-2511** (Apache, takes 3 reference
  images — species photo + style exemplar) = local Nano-Banana-equivalent
  for marking corrections.
- **API routes**: hero art / QC failures → **Nano Banana Pro batch**
  ($0.067/img, commercial outputs, SynthID-watermarked) or **Recraft custom
  style** (persistent trained style_id, ~$0.044/img, full ownership).
- License landmines confirmed: FLUX.1-dev/FLUX.2-dev still non-commercial;
  Sana weights research-only; SD3.5 fine under $1M but no upside.

**Costs**: initial 1,400 species × 6 candidates all-local ≈ $0 (~7 GPU-hrs);
hybrid with 15% NBP ≈ $85. Quarterly churn ≈ $0–21 hybrid.

## Stage 4 — Output QA + closed loop

Judge = ensemble, never a lone VLM:
- **Style gate**: CSD style-embedding cosine to the approved-masters
  centroid (+ DINOv2 probe trained on accumulated accept/reject labels).
  VLMs provably cannot judge this (Gemma rated marten ≡ blue jay).
- **Artifact floor**: pyiqa (MUSIQ/TOPIQ) + PAL4VST artifact masks;
  preference scorers (PickScore/HPSv3) only as bottom-5% catastrophe floor.
- **Anatomy**: keypoint model re-run on the illustration + Qwen3-VL
  two-image check against the reference.

Escalation ladder per species (log every verdict):
1. best-of-N + rerank (day-one win)
2. deterministic reroll playbook (new seed / strength step / next photo)
3. targeted repair: defect bbox → feathered mask → differential-diffusion
   inpaint with the same LoRA (stays on-style); global drift → img2img
   re-pass @ 0.35–0.5
4. LLM re-prompt with feedback memory (Idea2Img pattern)
5. human queue (the hotkey UI) — flagged cases only

**Flywheel**: one verdict schema feeds three things — judge calibration,
custom defect-detector training, and at ~500 verdicts a **Diffusion-KTO**
pass (takes unpaired accept/reject) to tune the LoRA itself. Cap synthetic
training data at ~25% and anchor on human-curated masters (model-collapse
guard).

## What runs where

| Component | Runtime | Cost |
|---|---|---|
| Photo sourcing | offline S3/SPARQL batch | $0 |
| Input audit | MegaDetector + keypoints + Qwen3-VL-8B local | $0 (API assist ≤$150 once) |
| Generation | SDXL→Z-Image + LoRA + ControlNet, local | $0 |
| Repair | Qwen-Image-Edit local; NBP/Recraft for heroes | ~$85 initial, ~$21/qtr |
| Output QA | CSD + pyiqa + keypoints + Qwen3-VL local | $0 |
| Orchestration | Python loop (DSPy-style), verdict log | $0 |
| Design/debug | Claude (this) | expensive — not in the runtime loop |
