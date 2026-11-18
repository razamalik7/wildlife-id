# AnimlVision House Style — v1 spec

Agreed 2026-10-12 (reasoning and history: [STYLE_DISCUSSION.md](STYLE_DISCUSSION.md)).
This replaces the palette/outline sections of STYLE_BIBLE.md where they
conflict; STYLE_BIBLE.md still governs framing, vignettes and photo rules.

## The direction in one line

Clean stylized illustration with painted depth: each animal's own colors
pushed to their richest believable version, texture drawn as flat stylized
marks, posed like a living animal rather than a specimen.

## Reference set (13)

**Training set (12):** american marten, elk, caribou, axis deer,
white-tailed deer, rhesus macaque, mountain goat, nine-banded armadillo,
common box turtle, desert tortoise, snowy owl, harbor seal.

**Reference only (not trained on):** gila monster (color and pattern
intensity).

Top exemplars: marten, elk, box turtle.

## Rules

1. **Texture as flat stylized marks.** Fur strands, scales, plates and
   feathers are drawn as flat stylized shapes, not individually lit in 3D
   (marten fur, turtle legs, armadillo plates — not gila beads, nile
   monitor scales).
2. **Texture where it matters.** Concentrate marks at edges, focal areas
   (face, mane, chest, tail tip) and slightly denser at highlights; leave
   big body surfaces as calm painted color. Never the same mark repeated
   evenly over the whole body (snowshoe hare).
3. **Painted depth.** A clear lit side and shadow side, far limbs darker,
   shadow under the body, overlap where the animal allows (marten tail,
   gila tail curl). Built from distinct color steps, never airbrushed
   gradients (grizzly, polar bear) or blended mud (swan head).
4. **Color pushed, never invented.** Each species' true colors at their
   richest believable version; key markings and accents brought out (axis
   spots, turtle shell, elk mane, owl eye, macaque face). Neutral animals
   still get color temperature: warm lights, cool shadows.
5. **Crisp color steps and small highlights.** Distinct tones you can see
   start and stop; small highlights on eyes, nose, shell, wet skin.
6. **Clean outline, thin to medium.** Present all the way around, dark,
   about 2–3 px at 768 px. Heavy even outlines are the ceiling (polar bear,
   peregrine); box turtle and harbor seal are the heaviest acceptable.
7. **Living pose.** Standard poses are fine, but caught mid-movement or
   mid-attention; body turned into space rather than flat to the page when
   the animal allows; head turned toward the viewer or locked on something.
   No specimen display (spread-wing crane, pinned-flat anoles).
8. **A face that reads.** Visible eye with a highlight; expression natural.
   Friendliness capped at the marten (harbor seal is the cutest acceptable).
9. **Props quieter than the animal.** Vignettes (water, branch, rock) per
   STYLE_BIBLE, lower contrast and saturation than the animal.
10. **Always from a reference photo.** Anatomy and markings come from a
    vetted real photo; the style never invents anatomy.

## Known misses (use as negative examples)

| Miss | Examples |
|---|---|
| Textbook (hatching everywhere, drab, specimen pose) | snowshoe hare, cardinal, bobcat, swan, most lizards |
| Too flat | bullfrog, cane toad, tegu, flamingo, blue jay |
| Too 3D-rendered | tokay gecko, turkey, nile monitor, gila (as a training target) |
| Too detailed / dark | chameleon, starling |
| Too arty (loose brushwork) | iguana, sparrow, heron, moose |
| Sticker / vector | polar bear, grizzly |
| Dead | pelican, crane |

## Coverage gaps

The reference set has no amphibians, snakes, lizards, small birds or large
birds. Each needs 1–2 anchor pieces made to this spec before the style is
trained.
