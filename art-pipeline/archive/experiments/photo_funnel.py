"""Reference-photo selection funnel: detector -> CLIP -> local VLM.

Scores every candidate photo per species and reorders candidates.json so
the vetting UI shows the best first (and slot-1 can be auto-accepted at
scale). All stages run locally.

  B. MegaDetector: exactly one animal, bbox 15-75% of frame, not touching
     edges, crop sharpness. Hard-fails go to the bottom.
  C. CLIP: similarity to "sharp side view photo of a single adult {species},
     whole animal visible" minus drawing/hand/enclosure negatives.
  D. Gemma 3 (via Ollama) on the top N survivors: rubric JSON verdict
     (whole body visible, nothing in mouth, adult, pose). Skipped with
     --no-vlm if Ollama isn't running.

Usage:
    art-pipeline/lora/venv/Scripts/python art-pipeline/photo_funnel.py
        [--candidates art-pipeline/vetting/candidates.json]
        [--vlm-top 5] [--no-vlm] [--validate art-pipeline/vetting/picks_overnight.json]

Writes funnel_scores.json next to candidates.json and rewrites
candidates.json with per-species candidates sorted best-first
(each candidate gains a "funnel" block; original order kept as "orig_index").
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

OLLAMA = "http://localhost:11434"


def fetch_image(url: str, cache: Path, photo_id: int) -> Image.Image | None:
    p = cache / f"{photo_id}.jpg"
    if not p.exists():
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "animlvision"})
            r.raise_for_status()
            Image.open(BytesIO(r.content)).convert("RGB").save(p, quality=90)
        except Exception:
            return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def stage_b(det_model, im: Image.Image) -> dict:
    """MegaDetector: single animal, well-framed, sharp."""
    arr = np.array(im)
    res = det_model.single_image_detection(arr)
    dets = [
        (box, conf) for box, conf, cls in zip(
            res["detections"].xyxy, res["detections"].confidence,
            res["detections"].class_id)
        if conf > 0.30 and cls == 0  # class 0 = animal
    ]
    out = {"n_animals": len(dets)}
    if len(dets) != 1:
        out["hard_fail"] = "no animal" if not dets else "multiple animals"
        return out
    (x1, y1, x2, y2), conf = dets[0]
    W, H = im.size
    area_frac = ((x2 - x1) * (y2 - y1)) / (W * H)
    edge_touch = x1 < 0.01 * W or y1 < 0.01 * H or x2 > 0.99 * W or y2 > 0.99 * H
    crop = np.array(im.crop((x1, y1, x2, y2)).convert("L"), dtype=np.float64)
    # variance of Laplacian via numpy (avoid cv2 dependency)
    lap = (np.abs(np.diff(crop, 2, axis=0)).var() + np.abs(np.diff(crop, 2, axis=1)).var()) / 2
    out.update(area_frac=round(float(area_frac), 3), edge_touch=bool(edge_touch),
               sharpness=round(float(lap), 1), det_conf=round(float(conf), 3))
    if area_frac < 0.15:
        out["hard_fail"] = "animal too small in frame"
    elif area_frac > 0.85:
        out["hard_fail"] = "animal overfills frame"
    elif edge_touch:
        out["hard_fail"] = "animal cut off by image edge"
    elif lap < 20:
        out["hard_fail"] = "blurry"
    return out


def build_clip(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def stage_c(clip_pack, device, im: Image.Image, common_name: str) -> float:
    model, preprocess, tokenizer = clip_pack
    pos = [f"a sharp clear side view photograph of a single adult {common_name}, whole animal visible"]
    neg = ["a drawing or illustration of an animal",
           "an animal held in a human hand",
           "an animal in a cage or enclosure",
           "a distant blurry animal in a landscape"]
    with torch.no_grad():
        img_f = model.encode_image(preprocess(im).unsqueeze(0).to(device))
        txt_f = model.encode_text(tokenizer(pos + neg).to(device))
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        sims = (img_f @ txt_f.T).squeeze(0).float().cpu().numpy()
    return round(float(sims[0] - sims[1:].max()), 4)


def stage_d(im: Image.Image, common_name: str) -> dict | None:
    """Qwen3-VL attribute extraction via Ollama. The VLM reports ATTRIBUTES
    only; acceptance policy lives in stage_d_policy() — the 3-model shootout
    showed every VLM extracts pose/in-mouth reliably but none can hold our
    usability policy (all rated a flying duck 'usable')."""
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        f"Describe this photo of a {common_name} for an illustrator's records. "
        "Answer STRICT JSON only: {\"single_animal\": bool, \"whole_body_visible\": bool "
        "(all limbs, feet and tail in frame and unobstructed), \"nothing_in_mouth\": bool "
        "(no food or objects in mouth or paws), \"adult_typical\": bool (adult with typical "
        "coloration, not juvenile/leucistic), "
        "\"pose\": \"standing/sitting/swimming/flying/feeding/preening/curled/other\", "
        "\"side_or_three_quarter_view\": bool, "
        "\"is_photo\": bool (a real photo, not a drawing), \"usable_score\": 0-10}")
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": "qwen3-vl:8b", "stream": False, "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        }, timeout=180)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="art-pipeline/vetting/candidates.json")
    ap.add_argument("--vlm-top", type=int, default=5)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--validate", default=None,
                    help="picks json to compare funnel top-1 against")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from PytorchWildlife.models import detection
    # MDV5A: well-calibrated confidences (0.9+ on clear animals). The compact
    # MDV6-yolov9-c variant runs badly under-confident on 500px thumbs and
    # missed 7/12 obvious animals in testing.
    det_model = detection.MegaDetectorV5(device=device, pretrained=True, version="a")
    clip_pack = build_clip(device)

    cand_path = Path(args.candidates)
    data = json.loads(cand_path.read_text())
    cache = cand_path.parent / "funnel_cache"
    cache.mkdir(exist_ok=True)
    scores_out = {}

    for slug, sp in data.items():
        rows = []
        for i, c in enumerate(sp["candidates"]):
            im = fetch_image(c["medium_url"], cache, c["photo_id"])
            if im is None:
                rows.append({**c, "orig_index": i, "funnel": {"hard_fail": "download failed", "score": -99}})
                continue
            b = stage_b(det_model, im)
            f = {"detector": b}
            if "hard_fail" in b:
                f["score"] = -10.0
            else:
                f["clip"] = stage_c(clip_pack, device, im, sp["common_name"])
                f["score"] = f["clip"] * 10
            rows.append({**c, "orig_index": i, "funnel": f, "_im": im})

        # VLM pass on the current top N
        if not args.no_vlm:
            rows.sort(key=lambda r: r["funnel"]["score"], reverse=True)
            for r in rows[: args.vlm_top]:
                if "_im" in r and "hard_fail" not in r["funnel"]:
                    v = stage_d(r["_im"], sp["common_name"])
                    if v:
                        r["funnel"]["vlm"] = v
                        # policy in code, attributes from the model
                        reject = (v.get("pose") in ("flying", "feeding", "preening")
                                  or not v.get("single_animal", True)
                                  or not v.get("nothing_in_mouth", True)
                                  or not v.get("is_photo", True))
                        if reject:
                            r["funnel"]["score"] -= 25
                            r["funnel"]["policy_reject"] = True
                        else:
                            checks = sum(bool(v.get(k)) for k in
                                         ["whole_body_visible", "adult_typical",
                                          "side_or_three_quarter_view"])
                            r["funnel"]["score"] += 2 * checks + float(v.get("usable_score", 0)) / 2

        rows.sort(key=lambda r: r["funnel"]["score"], reverse=True)
        for r in rows:
            r.pop("_im", None)
        sp["candidates"] = rows
        scores_out[slug] = [{"photo_id": r["photo_id"], "score": r["funnel"]["score"],
                             "fail": r["funnel"].get("detector", {}).get("hard_fail")}
                            for r in rows]
        top = rows[0]
        print(f"{slug}: top photo {top['photo_id']} score {top['funnel']['score']:.1f} "
              f"(was index {top['orig_index']})", flush=True)

    (cand_path.parent / "funnel_scores.json").write_text(json.dumps(scores_out, indent=1))
    cand_path.write_text(json.dumps(data, indent=1))
    print("candidates.json reordered best-first; funnel_scores.json written")

    if args.validate:
        picks = json.loads(Path(args.validate).read_text())
        agree = sum(1 for slug, p in picks.items()
                    if data.get(slug, {}).get("candidates", [{}])[0].get("photo_id") == p.get("photo_id"))
        print(f"validation: funnel top-1 matches manual pick for {agree}/{len(picks)} species")


if __name__ == "__main__":
    main()
