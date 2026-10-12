# Style discussion log

Working notes from the style-direction conversation (2026-10-12). Nothing
here is decided yet; it records the user's groupings, Claude's close-up read
of them, and open questions.

## User's groupings (verbatim intent)

| User family | Pieces | User verdict |
|---|---|---|
| Ultra-stylized sticker | polar bear | could be sold |
| Realistic / stylized | gila monster, armadillo, nile monitor, ocelot | likes |
| Paintbrushy | gray wolf, rock pigeon, wolverine | likes |
| Textured, a bit sketchy, field-guide science | muskox, great blue heron | likes |
| Very flat cartoonish | grizzly bear | probably no |
| Cartoonish stylized | black bear, arctic fox, caribou, peregrine falcon | likes |
| Between stylized-realistic and stylized-cartoon | box turtle, elk, marten (top 3 favorites) | loves |
| Unplaced likes | groundhog, fox squirrel ("something a little off"), eastern newt, desert tortoise, snowy owl, harbor seal ("very cartoonish but not in a bad way") | likes |
| Cool but soulless / textbook | snowshoe hare, great blue heron, whooping crane, both anoles | dislikes the feeling |

Separate "life" axis the user raised: eye size, humanization, cuteness,
dramatization. Alive: ocelot (very), gila (color), nile monitor (detail/texture),
and most of the pieces named above.

## Claude's close-up read: two separate dimensions

**1. Technique (how it is drawn).** Five techniques, from close-up crops:

- **Graphic / vector**: polar bear (heavy even navy outline, cel shading with
  blue shadows), grizzly (smooth clip-art fills).
- **Clean stylized**: clean confident outline, color laid in as broad painted
  fields, texture only where it defines the animal (mane, tail tip, pattern).
  Marten, elk, box turtle, arctic fox, caribou, peregrine falcon, ocelot,
  black bear, harbor seal, eastern newt.
- **Painterly brush**: visible dabs and strokes building the fur/feathers,
  rim light. Gray wolf, rock pigeon, wolverine.
- **Pen-and-ink with color wash**: fine hatched pen lines everywhere over a
  color wash, the classic field-guide plate. Groundhog, fox squirrel, muskox,
  snowy owl, great blue heron, snowshoe hare.
- **Realistic rendered**: soft, photo-like modelling, fine scales/plates,
  little or no outline. Gila monster, armadillo, nile monitor, desert tortoise.

Notes:
- The user's three "loves" and all four "cartoonish stylized" pieces fall
  in the same technique (clean stylized). So do ocelot, harbor seal, newt.
- Clean stylized is the only technique that already spans every class among
  the user's likes: mammals, a bird (falcon), a reptile (box turtle), an
  amphibian (newt) and a pinniped (harbor seal).
- Within it, outline weight varies (turtle/seal bolder, fox lighter) and
  color intensity varies (elk/turtle/newt rich, fox/caribou muted).
- "Textbook" pieces are mostly pen-and-ink or realistic technique combined
  with a specimen pose and low color (hare, crane, anoles, heron).

**2. Life (how present the animal feels).** Independent of technique:

- **Pose**: caught mid-movement or mid-attention (wolf striding, ocelot
  stalking, caribou walking) vs. specimen display (crane wings fanned
  symmetrically, anoles pinned flat to a stick, hare in neutral profile).
- **Gaze / engagement**: head turned toward the viewer or locked on
  something (marten, elk, ocelot, falcon) vs. blank profile.
- **Drama**: strong value contrast, rim light, bold pattern (ocelot, turtle,
  falcon, muskox).
- **Character / humanization**: harbor seal (round eyes, soft smile line:
  cute), fox squirrel (frontal, symmetric, paws clasped like a mascot: likely
  the "something off").
- **Color**: rich color carries the stylized pieces (gila, newt, elk).

## Round 2: user's own groups (supersede Claude's five techniques)

| Group | Pieces | User notes |
|---|---|---|
| **Magic** | ocelot, marten, armadillo, gila monster, box turtle, nile monitor, elk (maybe) | |
| **Stylish** | axis deer, arctic fox, mountain goat, elk, caribou, peregrine falcon, black bear, marten, snowy owl, badger | like cartoony but less big-eyed / friendly |
| **Artistic** | wolverine, gray wolf, muskox, whooping crane, rock pigeon, groundhog, both anoles, eastern newt (maybe) | varies too much in color vibrance and texture right now |
| **Cartoony** | northern elephant seal, harbor seal | |

Also from the user:
- Pen-and-ink group makes sense; realistic-render "kind of"; ocelot arguably
  belongs with realistic render.
- Polar bear is close to stylish; what breaks it is the heavy border and the
  pale color-wash shading with no deep blacks.
- Pose matters a lot but isn't everything: jaguar (still not loved) and black
  bear (good pose, still not alive enough).
- Rhesus macaque feels very alive; unplaced.
- Wolf feels alive and artsy; wolverine a little less alive (maybe the animal).
- Ocelot isn't looking at the viewer yet is very alive; the marten looking at
  the viewer may help it.

Measured lean of each group vs. the whole collection (typical-spread units,
|x| >= 0.7 is a clear lean; small groups, supporting evidence only):

| | Magic | Stylish | Artistic | Cartoony | Marten + elk |
|---|---|---|---|---|---|
| Outline thickness | thin -0.9 | 0.0 | thin -0.9 | **thick +1.1** | thin -0.9 |
| Flat calm color areas | -0.3 | **+1.4** | -0.2 | +0.5 | **+1.9** |
| Crisp detail (hard edges / local contrast) | **+0.9 / +0.8** | -0.7 / -0.5 | ~0 | ~0 | **-1.6 / -1.3** |
| Drawn lines inside | +0.6 | -0.7 | +0.3 | -0.5 | -1.1 |
| Saturation | +0.2 | **-1.2** | -0.5 | -0.8 | +0.7 |
| Warmth | **+0.8** | -0.7 | -0.7 | -0.2 | **+1.0** |
| Rim light | 0.0 | 0.0 | **+0.8** | +0.6 | +0.5 |

Reading: Magic = crisp, detailed, warm, thin-lined. Stylish = calm flat
fields, muted, cooler (partly because several stylish animals are white).
Artistic = rim light and fine texture. Cartoony = thick outline. Marten and
elk (in both Magic and Stylish, and among the user's top favorites) are the
calmest, flattest pieces with Magic's warmth and color: Stylish structure,
Magic color.

Class coverage among the user's likes:
- Magic: mammals + reptiles (no birds, amphibians, pinnipeds)
- Stylish: mammals + birds (no reptiles, amphibians, pinnipeds)
- Artistic: mammals, birds, reptiles, amphibian (no pinnipeds)
- Cartoony: pinnipeds only

## Parked for later (do not drop)

- UX redesign of the app.
- Language / voice of the product now that it is a real product.
- Using the art more prominently across the product ("the art is the heart
  and soul").
- Measuring eyes properly (needs an eye locator, e.g. Qwen3-VL grounding,
  then code to measure size and catchlight).
- Measuring pose (animal keypoint model: head turn, stride).

## Open questions for the user

See the conversation; answers get recorded here.
