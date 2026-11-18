"""Test local Ollama VLMs (qwen3-vl:8b vs gemma3:12b) on:
1. Describing ONLY visible colors/markings on the actual image (using Wiki as background knowledge)
2. Detecting obscured/clipped anatomy (e.g. hooves in grass, webbed feet underwater, hind paws behind rock)
   and outputting BOTH:
   - a prompt clause describing the completed full-body anatomy
   - a structured 'completion_recipe' (e.g. 'hooves_below', 'duck_feet_tucked', 'duck_feet_paddling', 'hind_paws_below', 'none')
     so our deterministic image-prep step knows whether/where to seed latent space below the cutout!
"""

import base64
import json
import time
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image

OLLAMA = "http://localhost:11434"
WORK = Path(__file__).parent / "batch_run_v4"
WIKI_RAW = json.loads((Path(__file__).parent / "wiki_masters_raw.json").read_text(encoding="utf-8"))

TEST_PHOTOS = [
    ("mule_deer", "Mule Deer", WORK / "mule_deer_photo.png",
     "Mule deer have large ears, atan-brown coat, white rump patch, black-tipped tail, forked antlers in males, and slender legs with dark cloven hooves."),
    ("pond_slider", "Pond Slider", WORK / "pond_slider_photo.png",
     "Adult pond sliders have an oval olive-to-dark-brown carapace, yellow and green striped skin on head and legs, a red stripe behind the eye, and webbed clawed feet."),
    ("green-winged_teal", "Green-winged Teal", WORK / "green-winged_teal_photo.png",
     "Breeding male green-winged teal have a chestnut head with green eye-patch, speckled buff breast, finely barred grey flanks, yellow undertail patch, and dark grey-brown webbed feet."),
    ("american_pika", "American Pika", WORK / "american_pika_photo.png",
     "American pikas are small, round, tailless lagomorphs with grizzled greyish-brown and buff fur, rounded ears, and four short furry legs with small paws."),
    ("muskrat", "Muskrat", WORK / "muskrat_photo.png",
     "Muskrats are medium-sized semiaquatic rodents with dense dark brown fur, rounded body, short front legs, webbed hind feet, and a long laterally flattened scaly black tail."),
]

PROMPT_TEMPLATE = """You are a wildlife illustration director inspecting a reference photo of a {common_name}.
Background species reference: {wiki_hint}

Carefully inspect THIS specific photo and return STRICT JSON only with these fields:
{{
  "pose_and_view": "short phrase describing the animal's pose and angle (e.g. 'standing side three-quarter view', 'floating side profile view')",
  "visible_colors_and_markings": "20-30 word comma-separated visual description of ONLY the true colors, patterns, and markings actually visible on this individual animal. Do NOT use ambiguous bleeding color words like 'pinkish' unless truly bright pink; do NOT describe juvenile colors if this individual is a dark adult.",
  "obscured_anatomy": "list any body parts hidden by water, grass, rocks, or framing (e.g. 'lower legs and hooves hidden in grass', 'webbed feet underwater', 'hind paws behind rock', 'long scaly tail underwater', or 'none')",
  "completed_anatomy_clause": "short phrase describing how the missing parts should appear in a complete full-body illustration (e.g. 'with all four slender legs and dark cloven hooves visible', 'with dark grey webbed feet tucked beneath belly', 'with rounded hind haunch and four small paws visible')",
  "completion_mode": "one of: ['none', 'extend_hooves_down', 'duck_feet_tucked', 'duck_feet_paddling', 'round_bottom_and_paws', 'extend_swimming_limbs']"
}}"""


def call_vlm(model: str, im_path: Path, common_name: str, wiki_hint: str) -> tuple[float, dict | str]:
    im = Image.open(im_path).convert("RGB")
    im.thumbnail((768, 768), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = PROMPT_TEMPLATE.format(common_name=common_name, wiki_hint=wiki_hint)
    t0 = time.time()
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        },
        timeout=180,
    )
    dt = time.time() - t0
    r.raise_for_status()
    content = r.json()["message"]["content"]
    try:
        return dt, json.loads(content)
    except Exception:
        return dt, content


def main() -> None:
    for model in ["qwen3-vl:8b", "gemma3:12b"]:
        print(f"\n==================== MODEL: {model} ====================")
        for slug, common, p, wiki in TEST_PHOTOS:
            dt, res = call_vlm(model, p, common, wiki)
            print(f"\n[{slug}] ({dt:.1f}s):")
            print(json.dumps(res, indent=2) if isinstance(res, dict) else res)


if __name__ == "__main__":
    main()
