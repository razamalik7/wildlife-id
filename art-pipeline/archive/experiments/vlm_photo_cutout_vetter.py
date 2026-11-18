"""Side-by-Side [Original Photo | BiRefNet Cutout] VLM Vetter & Trait Extractor (`gemma3:27b`).

Builds a 1024x512 composite showing:
  - Left half (512x512): Original uncropped reference photograph (with a red border)
  - Right half (512x512): BiRefNet segmented cutout on solid white (with a blue border)

And asks `gemma3:27b` to evaluate BOTH views simultaneously so it catches:
  1. `no_human_hands_in_photo`: Rejects any photo where the animal is held in a human hand/fingers/palm
     (even if BiRefNet cropped out most of the hand).
  2. `all_feet_and_toes_clearly_visible`: Rejects any photo/cutout where feet, paws, or bird toes are
     buried in grass, cut off at the bottom, or missing from the cutout.
  3. `no_grass_or_twigs_crossing_body`: Rejects any photo where a blade of grass, reed, or twig crosses
     in front of the animal's body/shell/tail.
  4. `no_branch_or_mud_stuck_to_cutout`: Rejects any cutout where a perch stick, rock, mud clump, or
     finger shadow remained attached to the animal's feet or body in the right-hand cutout panel.
"""

import base64
from io import BytesIO
import json
import requests
from PIL import Image, ImageDraw

OLLAMA_URL = "http://localhost:11434"


def build_side_by_side_vlm_panel(orig_im: Image.Image, cutout_im: Image.Image) -> Image.Image:
    """Create a 1024x512 side-by-side image: [Original Photo (Left) | BiRefNet Cutout on White (Right)]."""
    panel = Image.new("RGB", (1024, 512), (245, 245, 245))

    # Left: aspect-preserved original photo in 512x512
    s_o = min(504 / orig_im.width, 504 / orig_im.height)
    ow, oh = max(1, round(orig_im.width * s_o)), max(1, round(orig_im.height * s_o))
    orig_r = orig_im.resize((ow, oh), Image.LANCZOS)
    panel.paste(orig_r, ((512 - ow) // 2, (512 - oh) // 2))

    # Right: 512x512 normalized cutout
    cut_r = cutout_im.resize((504, 504), Image.LANCZOS)
    panel.paste(cut_r, (512 + 4, 4))

    draw = ImageDraw.Draw(panel)
    draw.rectangle([1, 1, 510, 510], outline=(200, 40, 40), width=2)
    draw.rectangle([513, 1, 1022, 510], outline=(30, 90, 200), width=2)
    return panel


def vet_photo_and_cutout_gemma27b(
    orig_im: Image.Image,
    cutout_im: Image.Image,
    common_name: str,
    sci_name: str,
    subtype: str,
    wiki_clauses: str,
) -> dict:
    """Evaluate [Original Photo | Cutout] side-by-side with strict occlusion/hand/foot vetoes."""
    panel = build_side_by_side_vlm_panel(orig_im, cutout_im)
    buf = BytesIO()
    panel.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    limb_rule = (
        "All legs, feet, paws, or bird toes must be 100% visible and intact in the RIGHT cutout (not hidden in grass, not cut off, and not missing toes)."
        if subtype in ("mammal", "bird", "lizard", "amphibian")
        else "The entire body, shell/fins, head, and tail must be 100% visible and unobstructed in the RIGHT cutout."
    )

    prompt = (
        f"You are inspecting a side-by-side pair for a {common_name} ({sci_name}) field-guide illustration:\n"
        f"- LEFT panel (red border): Original wild photograph.\n"
        f"- RIGHT panel (blue border): Segmented cutout on a white background.\n"
        f"Wikipedia reference description: {wiki_clauses}\n\n"
        f"Mandatory quality rules ({limb_rule}):\n"
        f"Return STRICT JSON only with these exact keys:\n"
        f"1. \"no_human_hands_in_photo\": boolean (false if a human hand, fingers, palm, or person is holding or touching the animal in the LEFT photo)\n"
        f"2. \"all_feet_and_toes_clearly_visible\": boolean (false if any foot, paw, or bird toe is hidden in grass/water, cut off, or missing in the RIGHT cutout)\n"
        f"3. \"no_grass_or_twigs_crossing_body\": boolean (false if a grass blade, reed, stem, or twig crosses over the animal's body, shell, or tail)\n"
        f"4. \"no_branch_or_mud_stuck_to_cutout\": boolean (false if a perch branch, wire, mud clump, rock, or finger shadow is still attached to the animal in the RIGHT cutout)\n"
        f"5. \"natural_unwarped_pose\": boolean (false if the animal is curled in a bizarre defensive knot, dead, or upside-down)\n"
        f"6. \"pose_quality_score\": number from 0 to 10 (10 = textbook full-body side/3/4 profile with clean feet and zero occlusion)\n"
        f"7. \"facing_direction\": \"left\" or \"right\"\n"
        f"8. \"visual_traits\": string of 18 to 30 words listing the specific colors, markings, patterns, and anatomical features of this {common_name} (combining the photo with the Wikipedia description; comma-separated phrases, no background words)."
    )

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": "gemma3:27b",
            "stream": False,
            "keep_alive": "5m",
            "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        },
        timeout=120,
    )
    r.raise_for_status()
    res = json.loads(r.json()["message"]["content"])

    # Compute strict composite score with hard vetoes
    hard_vetoes = [
        not res.get("no_human_hands_in_photo", True),
        not res.get("all_feet_and_toes_clearly_visible", True),
        not res.get("no_grass_or_twigs_crossing_body", True),
        not res.get("no_branch_or_mud_stuck_to_cutout", True),
        not res.get("natural_unwarped_pose", True),
    ]
    res["passed_all_vetoes"] = not any(hard_vetoes)
    res["veto_penalty"] = -25.0 * sum(1 for v in hard_vetoes if v)
    return res
