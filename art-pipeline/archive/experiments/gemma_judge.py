"""Gemma-3 vision harnesses for the art factory.

Two roles, both via local Ollama (gemma3:12b):

  curate  - classify each master's sub-style. The user's taste anchors:
            soft matte painterly (marten, harbor seal, gila monster, elk,
            whooping crane, great horned owl) IN; bold-ink sticker look OUT.
            -> v3_curation.json
  judge   - score a generated image against the reference photo + rubric
            distilled from the user's round-2 review. Calibration mode
            compares judge scores to the user's recorded verdicts.
            -> prints/saves rubric JSON per image

Usage:
    python art-pipeline/gemma_judge.py curate
    python art-pipeline/gemma_judge.py calibrate   # vs review_round2.json
    (judge() is imported by art_factory.py)
"""

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

OLLAMA = "http://localhost:11434"
MODEL = "gemma3:12b"
ROOT = Path(__file__).parent


def _b64(im: Image.Image, max_side=896) -> str:
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def ask(prompt: str, images: list[Image.Image], retries=2) -> dict | None:
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{OLLAMA}/api/chat", json={
                "model": MODEL, "stream": False, "format": "json",
                "options": {"temperature": 0.1},
                "messages": [{"role": "user", "content": prompt,
                              "images": [_b64(im) for im in images]}],
            }, timeout=300)
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"])
        except Exception:
            continue
    return None


CURATE_PROMPT = """You are curating a training set for an illustration style model.
The TARGET style is: soft matte painterly wildlife illustration - muted natural colors,
flat gouache-like paint, visible soft brushwork, restrained thin outlines or none,
NO gloss, NO heavy black sticker outline, NO oversaturated cartoon look, NO 3D shading.
Judge ONLY the rendering style of this illustration, not the animal species or pose.
Answer STRICT JSON:
{"matte_painterly": 0-10 (how strongly it matches the target style),
 "sticker_ink": 0-10 (heavy black outlines + punchy flat saturated fills),
 "glossy_3d": 0-10 (shiny highlights, airbrushed or 3D-render feel),
 "include": true/false (true only if it belongs in the TARGET style training set)}"""


def curate() -> None:
    masters = sorted((ROOT / "masters").glob("*.png"))
    out = {}
    for p in masters:
        v = ask(CURATE_PROMPT, [Image.open(p)])
        if v:
            out[p.stem] = v
            print(f"{p.stem}: matte={v.get('matte_painterly')} sticker={v.get('sticker_ink')} "
                  f"gloss={v.get('glossy_3d')} include={v.get('include')}", flush=True)
    (ROOT / "lora" / "v3_curation.json").write_text(json.dumps(out, indent=1))
    inc = [k for k, v in out.items() if v.get("include")]
    print(f"\nincluded: {len(inc)}/{len(out)}")


JUDGE_PROMPT = """Image 1 is a REFERENCE PHOTO of a {species}. Image 2 is a generated
illustration of the same animal that must satisfy a strict production rubric.
Style target: soft matte painterly - muted natural colors, flat gouache paint, soft
visible brushwork, thin restrained outline, no gloss/shine, no heavy black sticker
outline, not oversaturated, not 3D/airbrushed, not photorealistic.
Fidelity target: same markings/colors as the photo; do NOT invent features absent from
the photo; body complete (all limbs, feet, tail present, nothing cut off or missing);
exactly one animal; no background scenery except an optional small stylized water pool,
branch or rock; no speckles/confetti/noise; no text or watermark.
Answer STRICT JSON:
{"style_matte_painterly": 0-10, "too_glossy_or_ai": 0-10 (10 = very glossy/AI-looking),
 "heavy_sticker_outline": 0-10 (10 = worst), "oversaturated": 0-10 (10 = worst),
 "anatomy_matches_photo": 0-10, "invented_features": true/false,
 "body_complete": true/false, "single_animal": true/false,
 "artifacts": true/false (speckles/noise/glitches/extra objects),
 "overall_production_quality": 0-10}"""


def judge(generated: Path, reference: Path, species: str) -> dict | None:
    return ask(JUDGE_PROMPT.format(species=species),
               [Image.open(reference), Image.open(generated)])


def judge_score(v: dict) -> float:
    """Collapse rubric to one comparable number (higher = better)."""
    if not v:
        return -99.0
    s = (2.0 * v.get("style_matte_painterly", 0)
         + 1.5 * v.get("anatomy_matches_photo", 0)
         + 1.0 * v.get("overall_production_quality", 0)
         - 1.5 * v.get("too_glossy_or_ai", 0)
         - 1.0 * v.get("heavy_sticker_outline", 0)
         - 0.5 * v.get("oversaturated", 0))
    if not v.get("body_complete", True):
        s -= 15
    if not v.get("single_animal", True):
        s -= 20
    if v.get("artifacts", False):
        s -= 10
    if v.get("invented_features", False):
        s -= 5
    return round(s, 2)


def calibrate() -> None:
    """Score the round-2 outputs the user reviewed; print vs their verdicts."""
    review = json.loads((ROOT / "batch_run" / "review_round2.json").read_text())
    work = ROOT / "batch_run"
    print(f"{'species':28s} {'verdict':16s} {'@0.6':>7} {'@0.7':>7}")
    for slug, r in review.items():
        row = [slug[:27], r["verdict"][:15]]
        for s in ["0.6", "0.7"]:
            d = work / f"{slug}_str{s}"
            files = sorted(d.glob("*.png")) if d.exists() else []
            if files:
                v = judge(files[0], work / f"{slug}_photo.png", slug.replace("_", " "))
                row.append(f"{judge_score(v):7.1f}")
            else:
                row.append("      -")
        print(" ".join(str(x) for x in row), flush=True)


if __name__ == "__main__":
    {"curate": curate, "calibrate": calibrate}[sys.argv[1]]()
