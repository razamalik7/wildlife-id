# Style Fields — what to look at, in plain terms

13 things you can check on any illustration. For each: what it means, and
the options. The draft labels for all 47 superset pieces are in
[style_labels.csv](../archive/experiments/style_labels.csv) (review page: `localhost:8777/style.html`).
Some fields also have a **measured** number from [style_audit.py](../archive/experiments/style_audit.py),
so they don't rely on anyone's eye.

| # | Field | What to look at | Options |
|---|---|---|---|
| 1 | **Outline weight** | The line drawn around the edge of the animal. How thick is it? | none · thin · medium · heavy |
| 2 | **Outline color** | Is that edge line pure black, or a dark version of the animal's own color (dark brown on a deer, dark gray on a wolf)? | black · tinted · none |
| 3 | **Outline consistency** | Is the line the same thickness all the way around (sticker look), or does it thicken in shadows and thin/vanish on lit edges (hand-drawn look)? | uniform · varied |
| 4 | **Interior lines** | Lines drawn *inside* the animal — individual fur strokes, feather edges, scale outlines, wrinkles. | sparse · moderate · dense |
| 5 | **Shading method** | How shadows are made. *Cel* = 2–3 hard-edged tone steps, like a cartoon. *Painted* = soft brushy blends with visible strokes. *Rendered* = many smooth tones, close to a photo. | cel · painted · rendered |
| 6 | **Shadow depth** | How dark the darkest shadows get compared with the animal's base color. | light · medium · deep |
| 7 | **Saturation** | How intense the colors are. Muted = earthy, slightly grayed. Vivid = punchy. | muted · natural · vivid |
| 8 | **Texture density** | How much small detail overall — big simple color shapes vs. every hair and scale drawn. | low · medium · high |
| 9 | **Silhouette edge** | The outer shape: smooth and clean, or broken up by fur tufts and feather tips? | clean · textured |
| 10 | **Eyes** | Small and naturalistic, or big with bright highlights? (Owls and seals are naturally large.) | small · medium · large |
| 11 | **View / pose** | Which way the body faces. | profile · 3/4 · front · climbing · coiled … |
| 12 | **Prop** | Anything that isn't the animal. | none · water · branch · rock · stump |
| 13 | **Overall feel** | The gut-check category. *Field guide* = naturalist illustration. *Sticker* = bold even outline, graphic. *Near-real* = heavily rendered, almost a painting of a photo. | fieldguide · sticker · near-real |

Measured numbers (all 100 pieces, per animal, on the animal only):

- **flat_fraction** — share of the animal painted in calm, unbroken color areas
- **ink_fraction** — share of the animal that is near-black ink
- **detail** — how busy the texture is
- **saturation** — color intensity
- **outline_black** — share of the edge line that is pure black

## What your picks have in common (draft — verify on the review page)

Comparing your **22 ultra-favorites** against the other 25 in the superset:

| Field | Ultra 22 | Everything else |
|---|---|---|
| Overall feel | **22/22 field guide** | 5 near-real, 3 sticker |
| Shading | painted or cel, **never rendered** | 6 rendered |
| Outline | thin, varied, mostly **tinted** (17/22) | 2 heavy, 5 uniform, 10 black |
| Shadow depth | mostly light/medium (deep: 3/22) | deep: 9/25 |
| Props | **none** (1 rock) | branches, stump, water |
| Measured: flat color areas | **17.5%** of the animal | 12% |
| Measured: black ink | **4%** | 5–7% |

**Draft house-style rule from this:** a thin, hand-varied outline tinted to the
animal's own dark color; soft painted shading in a few steps, never
photo-rendered; shadows that stop at medium; earthy, slightly muted color;
texture *suggested* by a handful of strokes rather than drawn element-by-element;
big calm color areas.

## The cross-class problem

Per class in the current superset:

- **Mammals** lean sparse interior lines and low texture (10/26 sparse, 10/26 low).
- **Birds** lean moderate-to-dense lines (8/9), because feathers get drawn as
  individual edged shapes.
- **Reptiles** lean moderate-to-dense (9/10), because scales get outlined.

So birds and reptiles currently read busier and more "illustrated" than the
mammals beside them. For a Dex where an owl sits next to a fox next to a
turtle, the fix is a single **detail budget** that applies to every class:

- Feathers, scales and fur are **suggested** — a few strokes in the shadow
  areas and along edges — never every feather or scale outlined.
- Patterns (owl barring, turtle scutes, jaguar spots, rattlesnake diamonds)
  are painted as **color shapes**, not as lined shapes.
- The same outline weight, outline tint rule, shadow-depth cap and saturation
  band apply to every class. Species colors vary; the *rendering* doesn't.

Fields that must stay constant across ALL classes: 1, 2, 3, 4, 5, 6, 7, 8, 13.
Fields allowed to vary by species: color itself, 10 (eye size follows
anatomy), 11 (pose follows behavior), 12 (vignette rules in STYLE_BIBLE).
