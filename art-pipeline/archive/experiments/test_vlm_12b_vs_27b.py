"""Compare gemma3:12b vs gemma3:27b with strict NO-ENVIRONMENT rules on our 5 test species,
and test the resulting prompts directly through SDXL img2img on those 5 species first!
"""

import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image

OLLAMA = "http://localhost:11434"
WORK = Path(__file__).parent / "batch_run_v4"

TEST_PHOTOS = [
    ("mule_deer", "Mule Deer", WORK / "mule_deer_photo.png",
     "Mule deer have large ears, a tan-brown coat, white rump patch, black-tipped tail, forked antlers in males, and slender legs with dark cloven hooves."),
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

CRITICAL RULES:
1. Describe ONLY the animal's body, fur/feathers/scales/shell, and anatomy.
2. NEVER mention any environment, background, water, pond, grass, vegetation, leaves, rocks, logs, branches, sky, shadows, reflections, or wet sheen in `visible_colors_and_markings` or `completed_anatomy_clause`. The final illustration is isolated on a pure white background with zero props.
3. Describe ONLY the true colors visible on THIS individual animal (e.g., if an adult turtle shell is dark brown/black, say dark brown, NOT bright green; never use bleeding color words like 'pinkish').

Return STRICT JSON only with these exact fields:
{{
  "pose_and_view": "short phrase describing the animal's pose and angle without mentioning props (e.g. 'standing side three-quarter view', 'floating side profile view', 'crouching side three-quarter view')",
  "visible_colors_and_markings": "15-25 word comma-separated list of ONLY the animal's body colors, coat/plumage/shell patterns, and anatomical markings visible on this individual. Zero environment or background words.",
  "obscured_anatomy": "brief note of which body parts are hidden in the photo (or 'none')",
  "completed_anatomy_clause": "short phrase describing the completed body parts in isolation with zero environment words (e.g. 'with all four slender legs and dark cloven hooves visible', 'with dark grey webbed feet tucked beneath belly', 'with rounded hind haunch and four small paws visible', 'with webbed hind feet and long laterally flattened black tail visible')",
  "completion_mode": "one of: ['none', 'extend_hooves_down', 'duck_feet_tucked', 'duck_feet_paddling', 'round_bottom_and_paws', 'extend_swimming_limbs']"
}}"""

# Deterministic safety filter just in case any VLM ever slips an environment word through
ENV_BAN_WORDS = re.compile(
    r"\b(water|pond|lake|river|stream|grass|vegetation|leaves|leaf|plant|plants|rock|rocks|boulder|log|branch|twig|sky|blue water|reflection|reflections|wet sheen|wet|mud|dirt|ground|snow|sand)\b",
    re.I,
)


def sanitize_clause(text: str) -> str:
    parts = [p.strip() for p in text.split(",")]
    clean = [p for p in parts if p and not ENV_BAN_WORDS.search(p)]
    return ", ".join(clean)


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
        data = json.loads(content)
        data["visible_colors_and_markings"] = sanitize_clause(data.get("visible_colors_and_markings", ""))
        return dt, data
    except Exception:
        return dt, content


def main() -> None:
    out_all = {}
    for model in ["gemma3:12b", "gemma3:27b"]:
        print(f"\n==================== MODEL: {model} ====================")
        out_all[model] = {}
        for slug, common, p, wiki in TEST_PHOTOS:
            dt, res = call_vlm(model, p, common, wiki)
            out_all[model][slug] = res
            print(f"\n[{slug}] ({dt:.1f}s):")
            print(json.dumps(res, indent=2) if isinstance(res, dict) else res)
    (WORK / "vlm_12b_vs_27b.json").write_text(json.dumps(out_all, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
