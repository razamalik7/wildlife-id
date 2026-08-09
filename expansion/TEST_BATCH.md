# First-20 Expansion Test Batch

Chosen to stress every part of the new pipelines before scaling to
hundreds of species. Full data (taxon ids, observation counts, CC0/CC-BY
reference-photo candidates) in [test_batch_20.json](test_batch_20.json);
photo candidates also appended to
[../art-pipeline/references.csv](../art-pipeline/references.csv).
All counts are US research-grade wild observations.

## What each pick tests

**Duck cluster — batch-by-cluster art generation + dense CV neighborhood**
(joins the existing Mallard; one session, relative proportions enforced):

| Species | Obs |
|---|---|
| Wood Duck | 82,288 |
| Northern Shoveler | 65,347 |
| American Wigeon | 55,547 |
| Green-winged Teal | 42,477 |
| Northern Pintail | 32,709 |

**Deliberate confusables with the current 100 — hard-negative + geo-prior tests:**

| Species | Obs | Confusable with | Why it's a good test |
|---|---|---|---|
| Mule Deer | 153,104 | White-tailed Deer, Elk | classic overlap-range pair |
| Pond Slider | 175,881 | Painted Turtle | ubiquitous basking-turtle confusion |
| Gopher Snake | 74,935 | W. Diamondback Rattlesnake | an actual rattlesnake mimic |
| Gopher Tortoise | 30,946 | Desert Tortoise | near-identical, **disjoint ranges** — pure geo-prior test |
| American Toad | 149,125 | Cane Toad | native-vs-invasive distinction, product-relevant |
| Muskrat | 28,739 | Beaver, Nutria | completes THE aquatic-rodent trio |
| California Ground Squirrel | 58,592 | squirrel cluster | dense-neighborhood add |

**Cheap iconic adds — most-observed gaps, new art body plans:**

| Species | Obs | Note |
|---|---|---|
| American Robin | 333,269 | most-observed US bird, period |
| Western Fence Lizard | 190,393 | most-observed US reptile |
| Great Egret | 213,733 | white-plumage rendering test |
| Downy Woodpecker | 143,934 | new pose convention: clinging |
| Common Watersnake | 75,949 | swimming/coiled snake variety |
| E. Red-backed Salamander | 72,231 | tiny elongate body plan |
| Spring Peeper | 47,827 | tiny treefrog body plan |
| American Pika | 10,047 | iconic; lowest-data pick (still 10k obs) |

## Notes

- All 20 verified new by **taxon_id** (name matching is unreliable — iNat
  renamed bison to *Bos bison* and split the armadillo, which makes them
  look absent from the current config if you match on names).
- Reference-photo candidates were auto-picked by vote count with
  CC0/CC-BY license filter; verify pose/quality per the style bible
  criteria before generating (side/three-quarter, adult, whole animal).
