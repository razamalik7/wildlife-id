"""
Full 20-Species v2 Pipeline Rerun (`test_fresh_20_v2_full_rerun.py`):
Tests the 4 key upgrades across all 20 species (`01` through `20`):
1. Photo-Primary + Wiki-Assisted Trait Extraction (`Gemma3:27b`):
   - Uses the reference cutout photo as the primary ground truth for the individual's exact color morph and markings,
     using Wikipedia clauses ONLY when consistent with the photo (preventing generic Wiki morph descriptions like
     "9 to 13 dark crossbands with white spots" from overriding a copper-striped individual).
2. False-Animal Compound Common-Name Sanitizer (`sanitize_sdxl_subject_noun`):
   - Strips misleading other-animal tokens from compound common names (e.g., "Southern Alligator Lizard" ->
     "slender Elgaria lizard", "Spiny Softshell" -> "smooth-shelled Apalone softshell turtle without scutes")
     so CLIP never hallucinates an alligator head or hard tortoise scutes.
3. Pre-Stylization Cutout Repair + Perch-Stub Preservation:
   - Uses the GenAI-repaired cutouts on the 7 problem species (`02`, `06`, `08`, `09`, `10`, `14`, `15`) and preserves
     natural short perch stubs under perched birds' gripping feet (`06_house_finch`, `07_mourning_dove`, `08_red_winged_blackbird`).
4. Canny-Guided Stylization (`No Depth` for non-snakes, preserving cutout/perch silhouette in post-cleanup).
"""
import base64
import json
import os
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

sys.path.insert(0, str(Path(__file__).parent))
from test_fresh_20_e2e import (
    FRESH_20_SPECIES as SPECIES_20,
    OLLAMA_URL,
    STYLE_V6,
    NEG_V6_UNIFIED,
    clean_post,
    encode_prompt_sdxl_multichunk,
    unload_ollama_model,
)
from test_pre_stylization_crop_repair import build_canny_from_repaired

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
GENAI_WORK = Path(__file__).parent / "batch_run_true_genai_repair"
V2_WORK = Path(__file__).parent / "batch_run_fresh_20_v2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

# False-animal tokens that poison SDXL's CLIP text encoder when they appear inside a different animal's common name
FALSE_ANIMAL_WORDS = {
    "alligator": {"lizard": "slender lizard"},
    "crocodile": {"lizard": "slender lizard", "newt": "salamander"},
    "leopard": {"frog": "spotted frog", "lizard": "spotted lizard", "shark": "spotted houndshark"},
    "tiger": {"salamander": "mole salamander", "beetle": "beetle", "shark": "requiem shark"},
    "elephant": {"seal": "marine pinniped seal", "shrew": "long-nosed sengi"},
    "spider": {"monkey": "primate monkey"},
    "kangaroo": {"rat": "desert rodent", "mouse": "desert mouse"},
    "bat": {"ray": "eagle ray"},
}


def sanitize_sdxl_subject_noun(common_name: str, sci_name: str, subtype: str) -> str:
    """Prevent CLIP token poisoning when a species' common name contains a different animal's name
    (e.g., 'Southern Alligator Lizard' -> 'slender Elgaria lizard', 'Spiny Softshell' -> 'smooth-shelled softshell turtle')."""
    lower = common_name.lower()
    genus = sci_name.split()[0] if sci_name else ""
    if "alligator" in lower and subtype == "lizard":
        return f"slender {genus} lizard ({sci_name})"
    if "softshell" in lower or subtype == "turtle":
        if "softshell" in lower:
            return f"smooth-shelled leathery softshell turtle ({sci_name}) with a flat un-scuted pancake shell"
    if "bat ray" in lower:
        return f"smooth-skinned Myliobatis eagle ray ({sci_name})"
    if "leopard shark" in lower:
        return f"slender Triakis houndshark ({sci_name})"
    return f"{common_name} ({sci_name})"


def query_gemma27b_photo_primary_traits(
    cutout_im: Image.Image,
    common_name: str,
    sci_name: str,
    subtype: str,
    wiki_clauses: str,
) -> str:
    """Extract 18-30 words of visual traits where the PHOTO CUTOUT is the primary ground truth for
    the individual's exact color morph/pattern, and Wikipedia only supplements consistent details."""
    buf = BytesIO()
    cutout_im.resize((512, 512), Image.Resampling.LANCZOS).save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f"You are describing a wildlife reference cutout of a { subtype } ({sci_name}) for a scientific illustrator.\n"
        f"Supplementary Wikipedia notes: {wiki_clauses}\n\n"
        f"STRICT RULES:\n"
        f"1. Describe the EXACT colors, pattern, markings, head shape, and surface texture visible on THIS specific individual photo FIRST.\n"
        f"2. Use the Wikipedia notes ONLY when they match what is visible in the photo (for example, to clarify smooth leathery shell vs scutes, or eye/bill color). If Wikipedia mentions a different color morph or markings (such as crossbands or white spots) that are NOT prominently visible on this specific individual, DO NOT include them.\n"
        f"3. NEVER use misleading words of other animals (such as 'alligator', 'crocodile', 'tortoise scutes').\n"
        f"Return STRICT JSON with a single key:\n"
        f"{{\"visual_traits\": \"18 to 30 comma-separated descriptive words of visible colors, patterns, and anatomy\"}}"
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "gemma3:27b",
                "stream": False,
                "keep_alive": "5m",
                "format": "json",
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            },
            timeout=90,
        )
        r.raise_for_status()
        data = json.loads(r.json()["message"]["content"])
        traits = str(data.get("visual_traits", "")).strip()
        if traits:
            return traits
    except Exception as e:
        print(f"  [Gemma v2 trait warning for {common_name}]: {e}", flush=True)
    return f"natural {sci_name} coloration and markings matching the reference photo"


def get_best_cutout_for_species(stem: str) -> Image.Image:
    """Return the GenAI-repaired cutout for the 7 repaired species, or the normalized cutout for the rest."""
    unified_p = GENAI_WORK / f"unified_cutout_{stem}.png"
    if unified_p.exists():
        return Image.open(unified_p).convert("RGB")
    return Image.open(WORK / "cutouts_1024" / f"{stem}.png").convert("RGB")


def main() -> None:
    V2_WORK.mkdir(parents=True, exist_ok=True)
    cut_v2_dir = V2_WORK / "cutouts_v2"
    canny_v2_dir = V2_WORK / "canny_v2"
    raw_v2_dir = V2_WORK / "raw_out"
    clean_v2_dir = V2_WORK / "clean_out"
    for d in (cut_v2_dir, canny_v2_dir, raw_v2_dir, clean_v2_dir):
        d.mkdir(parents=True, exist_ok=True)

    meta = json.loads((WORK / "fresh_20_metadata.json").read_text(encoding="utf-8"))
    traits_v2_path = V2_WORK / "traits_v2.json"
    traits_v2 = json.loads(traits_v2_path.read_text(encoding="utf-8")) if traits_v2_path.exists() else {}

    # Stage 1: Prepare all 20 cutouts & extract Photo-Primary + Wiki-Assisted visual traits via Gemma3:27b
    need_ollama = False
    for stem, common, sci, taxon_id, cls_name, subtype in SPECIES_20:
        cut_im = get_best_cutout_for_species(stem)
        cut_im.save(cut_v2_dir / f"{stem}.png")
        canny_im = build_canny_from_repaired(cut_im)
        canny_im.save(canny_v2_dir / f"{stem}.png")

        if stem not in traits_v2:
            need_ollama = True
            wiki_clauses = meta.get(stem, {}).get("wiki_clauses", "")
            t_str = query_gemma27b_photo_primary_traits(cut_im, common, sci, subtype, wiki_clauses)
            traits_v2[stem] = t_str
            print(f"[Traits v2 {stem}] {t_str}", flush=True)
            traits_v2_path.write_text(json.dumps(traits_v2, indent=2), encoding="utf-8")

    if need_ollama:
        unload_ollama_model("gemma3:27b")
        torch.cuda.empty_cache()

    # Stage 2: Load SDXL + Canny ControlNet + animl_style_v6
    print("Loading SDXL + Canny ControlNet + animl_style_v6...", flush=True)
    cnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0-small",
        torch_dtype=torch.bfloat16,
        variant="fp16",
    )
    cnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small",
        torch_dtype=torch.bfloat16,
        variant="fp16",
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        str(LORA / "base" / "sd_xl_base_1.0.safetensors"),
        controlnet=MultiControlNetModel([cnet_canny, cnet_depth]),
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(LORA / "output"), weight_name="animl_style_v6-step00001250.safetensors")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.to("cuda")

    for idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(SPECIES_20):
        cut_im = Image.open(cut_v2_dir / f"{stem}.png").convert("RGB")
        canny_im = Image.open(canny_v2_dir / f"{stem}.png").convert("RGB")
        subject_noun = sanitize_sdxl_subject_noun(common, sci, subtype)
        t_str = traits_v2.get(stem) or meta.get(stem, {}).get("visual_traits", "")

        prompt = (
            f"animl_gouache_style, a {subject_noun}, full body wildlife illustration, "
            f"{t_str}, {STYLE_V6}"
        )
        neg_prompt = NEG_V6_UNIFIED
        if stem == "15_spiny_softshell":
            neg_prompt = "scutes, tortoise shell plates, polygonal shell grid, hexagonal shell plates, " + neg_prompt
        elif stem == "14_southern_alligator_lizard":
            neg_prompt = "alligator head, crocodile snout, caiman, osteoderms, crocodilian, " + neg_prompt

        pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_sdxl_multichunk(
            pipe, prompt, neg_prompt, max_chunks=3
        )

        # Use slightly tighter denoise on lizards/turtles/snakes so exact photo markings & head shape stay 100% faithful!
        if subtype == "snake":
            strength = 0.48
            c_scale = [0.50, 0.0]
        elif stem in ("14_southern_alligator_lizard", "15_spiny_softshell"):
            strength = 0.50
            c_scale = [0.46, 0.0]
        else:
            strength = 0.56
            c_scale = [0.44, 0.0]

        print(f"[Stylizing v2 {idx+1}/20] {stem} (strength={strength}, noun='{subject_noun}')...", flush=True)
        gen = torch.Generator(device="cuda").manual_seed(42)
        res = pipe(
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            image=cut_im,
            control_image=[canny_im, canny_im],
            strength=strength,
            controlnet_conditioning_scale=c_scale,
            control_guidance_start=0.0,
            control_guidance_end=[0.74, 1.0],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
        ).images[0]

        out_raw = raw_v2_dir / f"{stem}.png"
        out_clean = clean_v2_dir / f"{stem}.png"
        res.save(out_raw)
        if out_clean.exists():
            out_clean.unlink()
        clean_post(res, out_clean)

        # Preserve repaired cutout silhouette / short perch stubs so BiRefNet never clips paws, toes, or short perches!
        clean_im = Image.open(out_clean).convert("RGBA")
        c_arr = np.array(clean_im)
        r_arr = np.array(res.convert("RGB"))
        rep_fg = np.any(np.array(cut_im) < 245, axis=-1).astype(np.uint8) * 255
        rep_fg = cv2.erode(rep_fg, np.ones((3, 3), np.uint8), iterations=1)
        rep_alpha = cv2.GaussianBlur(rep_fg.astype(np.float32), (5, 5), 1.0).astype(np.uint8)
        union_alpha = np.maximum(c_arr[:, :, 3], rep_alpha)
        c_arr[:, :, :3] = np.where(union_alpha[..., None] > 12, r_arr, 255)
        c_arr[:, :, 3] = union_alpha
        Image.fromarray(c_arr, "RGBA").save(out_clean)
        shutil.copyfile(out_clean, ARTIFACT_DIR / f"fresh20_v2_{stem}.png")

    # Build two 10-row x 4-column comparison sheets:
    # 1. Raw Reference Photo | 2. Final Cutout (with Perch/GenAI Repair) | 3. Previous v1 Output | 4. New v2 Output
    for part_idx, chunk in [(1, SPECIES_20[:10]), (2, SPECIES_20[10:])]:
        col_w = 330
        row_h = 350
        header_h = 54
        cols = 4
        rows = len(chunk)
        sheet = Image.new("RGB", (col_w * cols, header_h + row_h * rows), (246, 246, 246))
        draw = ImageDraw.Draw(sheet)
        try:
            f_hdr = ImageFont.truetype("arialbd.ttf", 14)
            f_sub = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            f_hdr = ImageFont.load_default()
            f_sub = f_hdr

        headers = [
            "1. Raw Reference Photo",
            "2. Cutout (with Perch / GenAI Repair)",
            "3. Previous v1 Output",
            "4. New v2 Output (Photo-Primary + Repair)",
        ]
        for c_idx, h_text in enumerate(headers):
            draw.text((c_idx * col_w + 10, 16), h_text, fill=(20, 20, 20), font=f_hdr)

        for r_idx, (stem, common, sci, taxon_id, cls_name, subtype) in enumerate(chunk):
            y0 = header_h + r_idx * row_h
            draw.line([(0, y0), (col_w * cols, y0)], fill=(200, 200, 200), width=1)
            t_short = (traits_v2.get(stem, "")[:88] + "...") if len(traits_v2.get(stem, "")) > 88 else traits_v2.get(stem, "")
            draw.text((10, y0 + 5), f"{stem[:2]}. {common} ({sci}) [{cls_name}/{subtype}] — {t_short}", fill=(15, 50, 100), font=f_sub)

            paths = [
                WORK / "raw_photos" / f"{stem}.jpg",
                cut_v2_dir / f"{stem}.png",
                WORK / "clean_out" / f"{stem}.png",
                clean_v2_dir / f"{stem}.png",
            ]
            for c_idx, p in enumerate(paths):
                if not p.exists():
                    continue
                im = Image.open(p)
                if im.mode == "RGBA":
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    bg.paste(im, mask=im.split()[3])
                    im = bg
                else:
                    im = im.convert("RGB")
                im_copy = im.copy()
                im_copy.thumbnail((300, 300), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (300, 300), (255, 255, 255))
                canvas.paste(im_copy, ((300 - im_copy.width) // 2, (300 - im_copy.height) // 2))
                sheet.paste(canvas, (c_idx * col_w + 15, y0 + 32))

        out_sheet = ARTIFACT_DIR / f"v6_fresh_20_v2_part{part_idx}.jpg"
        sheet.save(out_sheet, quality=92)
        print(f"Saved {out_sheet}!", flush=True)


if __name__ == "__main__":
    main()
