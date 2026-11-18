"""Draft style labels for the 47-piece superset (Claude's visual pass).

Field definitions live in STYLE_FIELDS.md. These labels are a first draft
from ~400px review sheets — the user corrects them, and the corrected set
becomes the spec for the target house style.

Writes art-pipeline/style_labels.csv (labels + measured metrics joined).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent

FIELDS = ["outline_weight", "outline_color", "outline_consistency", "interior_lines",
          "shading", "shadow_depth", "saturation", "texture", "silhouette_edge",
          "eyes", "view", "prop", "overall_feel"]

# slug: (outline_weight, outline_color, outline_consistency, interior_lines,
#        shading, shadow_depth, saturation, texture, silhouette_edge,
#        eyes, view, prop, overall_feel)
L = {
 "american_badger":       ("thin","tinted","varied","dense","painted","medium","muted","high","textured","small","front-3/4","none","fieldguide"),
 "american_black_bear":   ("medium","black","uniform","moderate","rendered","deep","muted","high","textured","small","3/4","none","near-real"),
 "american_marten":       ("thin","tinted","varied","sparse","painted","light","natural","low","clean","small","3/4","none","fieldguide"),
 "arctic_fox":            ("thin","tinted","varied","sparse","painted","light","muted","low","clean","small","profile","none","fieldguide"),
 "axis_deer":             ("thin","tinted","varied","sparse","cel","medium","natural","low","clean","small","profile","none","fieldguide"),
 "california_sea_lion":   ("thin","black","uniform","sparse","rendered","deep","muted","low","clean","medium","3/4","none","near-real"),
 "caribou":               ("thin","tinted","varied","moderate","painted","medium","muted","medium","textured","small","profile","none","fieldguide"),
 "eastern_fox_squirrel":  ("thin","tinted","varied","sparse","painted","light","natural","low","clean","small","front-3/4","none","fieldguide"),
 "elk":                   ("thin","tinted","varied","moderate","painted","medium","natural","medium","clean","small","3/4","none","fieldguide"),
 "gray_wolf":             ("thin","black","varied","dense","painted","medium","muted","high","textured","small","3/4","none","fieldguide"),
 "groundhog":             ("thin","tinted","varied","moderate","painted","medium","muted","medium","textured","small","profile","none","fieldguide"),
 "harbor_seal":           ("thin","tinted","uniform","sparse","painted","medium","muted","low","clean","large","3/4","none","fieldguide"),
 "jaguar":                ("thin","black","varied","moderate","painted","medium","vivid","high","clean","small","3/4","none","fieldguide"),
 "mountain_goat":         ("thin","tinted","varied","moderate","painted","light","muted","medium","textured","small","profile","none","fieldguide"),
 "muskox":                ("thin","black","varied","dense","painted","deep","muted","high","textured","small","3/4","none","fieldguide"),
 "nine-banded_armadillo": ("thin","tinted","varied","moderate","painted","medium","muted","medium","clean","small","profile","none","fieldguide"),
 "northern_elephant_seal":("heavy","black","uniform","moderate","rendered","deep","muted","medium","clean","small","front-3/4","none","sticker"),
 "nutria":                ("thin","tinted","varied","sparse","painted","medium","natural","low","clean","small","profile","none","fieldguide"),
 "ocelot":                ("thin","black","varied","moderate","painted","medium","natural","high","clean","small","3/4","none","fieldguide"),
 "polar_bear":            ("heavy","tinted","uniform","moderate","cel","deep","muted","medium","clean","small","3/4","none","sticker"),
 "pronghorn":             ("thin","tinted","varied","sparse","cel","medium","natural","low","clean","small","profile","none","fieldguide"),
 "rhesus_macaque":        ("thin","tinted","varied","moderate","painted","medium","natural","medium","textured","small","front","none","fieldguide"),
 "snowshoe_hare":         ("thin","tinted","varied","sparse","painted","light","muted","low","clean","small","profile","none","fieldguide"),
 "white-tailed_deer":     ("thin","tinted","varied","sparse","cel","medium","natural","low","clean","small","3/4","none","fieldguide"),
 "yellow-bellied_marmot": ("thin","tinted","varied","moderate","painted","medium","natural","medium","textured","small","front","none","fieldguide"),
 "brown_pelican":         ("thin","black","varied","moderate","painted","medium","muted","medium","clean","small","profile","stump","fieldguide"),
 "great_blue_heron":      ("medium","black","varied","dense","painted","deep","natural","high","textured","small","profile","water","near-real"),
 "great_horned_owl":      ("medium","black","varied","dense","rendered","deep","natural","high","textured","large","front","branch","near-real"),
 "monk_parakeet":         ("thin","tinted","varied","sparse","painted","medium","vivid","low","clean","small","profile","none","fieldguide"),
 "osprey":                ("thin","black","varied","moderate","painted","deep","muted","medium","clean","small","3/4","branch","fieldguide"),
 "peregrine_falcon":      ("thin","black","varied","moderate","painted","medium","muted","medium","clean","small","3/4","rock","fieldguide"),
 "rock_pigeon":           ("thin","tinted","varied","moderate","painted","medium","muted","medium","clean","small","3/4","none","fieldguide"),
 "snowy_owl":             ("thin","tinted","varied","moderate","painted","light","muted","medium","clean","large","front","none","fieldguide"),
 "whooping_crane":        ("thin","black","varied","moderate","painted","medium","muted","medium","clean","small","front","none","fieldguide"),
 "wolverine":             ("thin","tinted","varied","moderate","painted","deep","muted","medium","textured","small","profile","none","fieldguide"),
 "american_alligator":    ("thin","tinted","varied","dense","rendered","deep","muted","high","clean","small","profile","none","near-real"),
 "brown_anole":           ("thin","tinted","varied","moderate","painted","medium","natural","medium","clean","small","climbing","branch","fieldguide"),
 "common_box_turtle":     ("medium","tinted","varied","moderate","painted","deep","vivid","medium","clean","small","3/4","none","fieldguide"),
 "desert_tortoise":       ("thin","tinted","varied","moderate","painted","light","muted","medium","clean","small","3/4","none","fieldguide"),
 "gila_monster":          ("none","none","none","sparse","painted","medium","vivid","medium","clean","small","profile","none","fieldguide"),
 "green_anole":           ("thin","tinted","varied","moderate","painted","medium","vivid","medium","clean","small","climbing","branch","fieldguide"),
 "green_iguana":          ("thin","tinted","varied","moderate","painted","medium","natural","medium","clean","small","profile","none","fieldguide"),
 "nile_monitor":          ("thin","tinted","varied","moderate","painted","medium","muted","medium","clean","small","upright","none","fieldguide"),
 "painted_turtle":        ("thin","tinted","varied","moderate","painted","medium","muted","medium","clean","small","3/4","none","fieldguide"),
 "western_diamondback_rattlesnake": ("thin","tinted","varied","dense","painted","medium","muted","high","clean","small","coiled","none","fieldguide"),
 "american_bullfrog":     ("thin","tinted","varied","moderate","painted","medium","natural","medium","clean","large","3/4","none","fieldguide"),
 "spotted_salamander":    ("medium","black","uniform","sparse","rendered","deep","vivid","low","clean","small","profile","none","sticker"),
}


def main() -> None:
    V = ROOT / "vetting"
    ultra = set(json.loads((V / "elite_ultra_v1.json").read_text())["keep"])
    core = set(json.loads((V / "elite_core_v1.json").read_text())["keep"])
    metrics = {r["slug"]: r for r in csv.DictReader(open(ROOT / "style_metrics.csv"))}
    cls = {s["name"].lower().replace(" ", "_"): s["taxonomy"]["class"]
           for s in json.loads((ROOT.parent / "frontend/public/species_config.json").read_text(encoding="utf-8"))}

    mcols = ["flat_fraction", "ink_fraction", "detail", "saturation", "outline_black"]
    with open(ROOT / "style_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slug", "class", "tier"] + FIELDS + [f"m_{c}" for c in mcols])
        for slug, labels in L.items():
            tier = "ultra" if slug in ultra else ("core" if slug in core else "superset")
            w.writerow([slug, cls.get(slug, "?"), tier, *labels] + [metrics[slug][c] for c in mcols])
    print(f"{len(L)} labelled -> style_labels.csv")


if __name__ == "__main__":
    main()
