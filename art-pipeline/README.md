# Art Pipeline

Everything needed to produce species illustrations at scale without style
drift. Art — not ML — is the bottleneck on expanding the Dex; this directory
is the fix.

## Contents

| File | Purpose |
|---|---|
| [STYLE_BIBLE.md](STYLE_BIBLE.md) | Production spec: line, palette, shading, framing. The contract. |
| [palette.json](palette.json) | K-means palette extracted from the live set (`extract_palette.py`). |
| [prompts/framing_contract.txt](prompts/framing_contract.txt) | Verbatim block appended to every generation prompt. |
| [reference-sheet/](reference-sheet/) | **Frozen** style references (6 pieces spanning body plans). Never update, never chain off new outputs. |
| [normalize.py](normalize.py) | Alpha-crop + rescale accepted art to a uniform 76%-fill 2048 canvas. |
| [extract_palette.py](extract_palette.py) | Regenerate palette.json if the accepted set materially changes. |

## Per-species generation procedure (near-term, prompt-based)

1. Pick a photo reference: side/three-quarter, adult, typical coloration, whole
   animal, even light. CC0 or CC-BY only (check the per-photo license on
   iNaturalist; many are all-rights-reserved). Log URL + license in
   `references.csv`.
2. Two-image prompt, explicit roles:
   > Match the illustration style of image 1 (style reference from
   > reference-sheet/, pick the closest body plan). Match the anatomy,
   > proportions, and markings of image 2 (photo). {species}, {one-line
   > diagnostic marks}.
   Then append `prompts/framing_contract.txt` verbatim.
3. Generate 4–6 candidates. Accept at most one (~30% accept rate is normal).
   Reject for: style drift, wrong diagnostic marks, prop/shadow leakage,
   non-left facing, cropped anatomy.
4. `python art-pipeline/normalize.py --in <accepted> --out <staging>`, move the
   normalized master into `art-pipeline/masters/`, then run
   `python art-pipeline/make_web_icons.py` to regenerate the WebP the app
   serves from `frontend/public/icons/web/`.
5. Batch by taxonomic cluster (all mustelids in one session, all dabbling
   ducks in another).

## The real fix: style LoRA (unblocks scale)

Plan, per the renovation context doc:

- Curate the existing ~101 pieces to the 40–60 most on-model. Train a style
  LoRA: 1000–2000 steps, LR ~1e-4, rank 16–32. Caption content only ("a red
  fox, side profile") — never describe the style in captions, or the model
  treats it as optional.
- Training data = illustrations + captions only. Photos are *not* training
  data. The ~20 existing photo/illustration pairs become the eval set:
  regenerate those species and compare against the handmade versions.
- Inference after LoRA: LoRA (style) + IP-Adapter/img2img photo reference
  (anatomy) + short prompt (framing contract). The style reference image
  drops out entirely. Main dial: IP-Adapter weight / denoise strength — too
  high bleeds photo texture, too low ignores real markings.

### Base model decision (checked 2026-08)

- **FLUX.1-dev is out for this product without a paid license**: the
  [non-commercial license v2.0](https://bfl.ai/legal/non-commercial-license-terms)
  allows commercial use of *outputs*, but self-hosting the model for a
  revenue-generating app and training/fine-tuning derivatives (a LoRA is a
  derivative) for commercial use both require a
  [paid commercial license](https://bfl.ai/licensing).
- **SDXL is the pragmatic default**: fully open license, mature tooling
  (kohya_ss / OneTrainer), trains faster and cheaper, and for a flat
  illustrated style the quality gap vs Flux is likely invisible.
- **RTX 5080 (Blackwell, sm_120) is fine now**: needs CUDA 12.8+ builds of
  PyTorch (current stable releases ship them; older wheels fail with
  "sm_120 unsupported"). SDXL LoRA at 1024px, fp16, batch 1–2: ~30–60 min per
  run. Expect 5–10 retrains while tuning — which is exactly why we train
  locally instead of renting.

### Ownership note (resolve before betting the brand on it)

US Copyright Office position: purely AI-generated images aren't
copyrightable — we may own the *right to use* the art while being unable to
stop copying. The trained LoRA itself is a more defensible asset than any
individual image. Option on the table: commission human illustration for a
flagship ~20 species (the screenshot species) and use them as the LoRA
training core.
