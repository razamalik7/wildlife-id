# AnimlVision Style Bible

Production spec for species illustrations. Drafted from direct inspection of the
existing set (fox, eagle, bison, rattlesnake, bullfrog, blue jay, painted turtle,
marten) plus k-means palette extraction over all 101 pieces ([palette.json](palette.json)).
Treat this as the contract another illustrator — human or model — could follow
without seeing the originals.

## Core specification

**Medium.** Digital gouache/ink hybrid. Color fields are built from visible,
directional brush strokes — never gradients, never airbrush, never smooth blends.
Every stroke follows the growth direction of fur, feathers, or scale rows.

**Outline.** Continuous exterior contour on the full silhouette. Weight ≈ 5–7 px
on a 2048 canvas (0.25–0.35% of canvas width). Color is a near-black warm brown
(`#1a1512`) or a much darker shade of the adjacent fill — never pure `#000000`,
never a cool black. Interior structure lines (leg separation, wing edge, shell
scutes) at roughly half exterior weight. No sketchy double lines.

**Shading.** Exactly 2–3 discrete value steps per color region, applied as
hard-edged painterly strokes. Light source upper-left, soft and unspecified.
No cast shadow, no ground shadow, no rim light, no specular shine on fur.

**Eyes.** Small and naturalistic — alert, not cute. Dark pupil, iris amber/brown
unless the species differs. At most one small soft highlight. No oversized
anime catchlights.

**Texture density.** Mid-level: individual feather groups and fur tufts are
drawn, individual hairs are not. The animal must still read cleanly at a 128 px
thumbnail. Reptile scales are drawn individually only where diagnostic (rattler
diamonds, turtle scutes); elsewhere implied by value bands.

**Anatomy.** Faithful and adult-typical. Field-guide accuracy on diagnostic
marks (eye-stripes, wing bars, dewlaps, rattle segments) — users learn what
species look like from this art. No chibi proportions, no anthropomorphic
expression, no dynamic action pose.

## Palette

Working palette extracted from the live set (share = fraction of all opaque
pixels). Earth-tone core; muted cool accents. Species-specific colors (cardinal
red, jay blue, anole green) are allowed at full local saturation but shaded
with warm darks from this table, not with gray or blue-black.

| Role | Hex | Share |
|---|---|---|
| Shadow/outline darks | `#080807` `#181613` `#2a241e` | 20% |
| Dark fur/feather | `#39322a` `#494337` `#5d5341` | 22% |
| Mid earth | `#6f6551` `#847861` `#938c7c` | 16% |
| Warm rust/tan | `#90633c` `#ac7a4b` `#77492d` `#593320` | 17% |
| Light tan/buff | `#b1946c` `#d7b483` `#d49856` | 8% |
| Cream/white | `#ada796` `#cec4ae` `#e1ddcc` `#f1f3eb` | 11% |
| Accent rust-red | `#bf4729` | 2% |
| Accent cool (water/sky species) | `#5c8ca2` `#34625f` `#8eb9c9` | 3% |

## Framing contract

The block in [prompts/framing_contract.txt](prompts/framing_contract.txt) is
appended verbatim to every generation prompt. Summary of the contract:

- Full body, every limb and the complete tail visible; nothing cropped.
- Three-quarter view, head toward viewer-left. (Most of the existing set faces
  left; left-facing is the convention going forward.)
- Neutral standing pose, feet on an invisible level ground line. Coiled pose
  for snakes. No perch, branch, rock, foliage, water, or any prop.
- Transparent background. No cast shadow, no ground plane, no vignette.
- Single adult animal in typical coloration. No text, border, or watermark.
- Subject scale is NOT the artist's problem: normalize.py crops to the alpha
  bounding box and rescales every piece so its long side fills 76% of the
  canvas. True body size lives in the species data (`body_length_cm`) and size
  comparisons are rendered from data, never from relative art scale.

## Reference discipline

- `reference-sheet/` holds the frozen style references spanning the body plans
  (canid, large ungulate, raptor, snake, turtle, frog, small passerine,
  small mustelid). **Never regenerate, never replace, never chain off a new
  output.** Style drift in the old pipeline came from referencing the newest
  generation instead of a frozen set.
- Every generation is a **two-image prompt with explicit roles**: "Match the
  illustration style of image 1. Match the anatomy, proportions, and markings
  of image 2 (photo)." The photo reference is mandatory for every species —
  without it, models draw a generic small brown mammal instead of an actual
  marten.
- Photo reference criteria: side or three-quarter view, adult, typical
  coloration, clear even light, entire animal in frame. Source from
  iNaturalist research-grade or Wikimedia, **CC0 or CC-BY only** (iNat photos
  carry per-photo licenses; many are all-rights-reserved). Record the source
  URL and license per photo in `references.csv`.

## Batch workflow

1. Batch by taxonomic cluster (all mustelids one session, all dabbling ducks
   another) — within-session consistency is free consistency, and relative
   proportions get enforced where confusion actually happens.
2. Generate 4–6 candidates per species; keep at most one. Expected accept rate
   ≈ 30%. Curation is where consistency comes from — a rejected candidate
   costs cents, an off-model accepted one degrades the whole set.
3. Run `normalize.py` on every accepted piece before it enters the app.

## Known deviations in the current set (do not imitate)

- Bald eagle: perched on a branch prop, near-pure-black heavy inking.
- Blue jay / small birds: subject fills as little as 25% of canvas (fixed by
  normalize.py going forward).
- Marten and some small mammals: outline nearly absent, softer rendering.
- Right-facing pieces exist; new work faces left. Existing art is NOT
  regenerated to comply — it gets revisited only after the style LoRA lands.
