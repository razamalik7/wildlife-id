"""
Fix 19_leopard_shark and 20_bat_ray:
1. 19_leopard_shark:
   - In v2, replacing 'Leopard Shark' with 'Triakis houndshark' caused CLIP to fall back to its 'Tiger Shark' prior (vertical stripes),
     and lowering Canny from 0.48 -> 0.44 loosened the spot boundaries!
   - Restoring 'Leopard Shark (Triakis semifasciata)' + v1's exact Wiki+Photo traits + Canny=0.48 locks the exact dotted saddle bars & spots.
2. 20_bat_ray:
   - The murky green underwater photo in v1 was selected BEFORE we built the side-by-side `[Original Photo | Cutout]` VLM vetter,
     and sunlight caustics on the murky cutout made v2's photo-primary trait extractor hallucinate spots on the ray!
   - Here we fetch clean Bat Ray (`Myliobatis californica`) candidates, vet `[Original Photo | Cutout]` side-by-side to reject murky/spotted-caustic underwater shots,
     and stylize with Wikipedia's true field mark ('smooth unspotted dark-brown dorsal surface, white underside').
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
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
    STYLE_V6,
    NEG_V6_UNIFIED,
    clean_post,
    download_pil,
    encode_prompt_sdxl_multichunk,
    fetch_inat_curated_taxon,
    fetch_inat_obs,
    fetch_wikimedia_commons,
    make_normalized_cutout,
)
from test_pre_stylization_crop_repair import build_canny_from_repaired

LORA = Path(__file__).parent / "lora"
WORK = Path(__file__).parent / "batch_run_fresh_20_e2e"
V2_WORK = Path(__file__).parent / "batch_run_fresh_20_v2"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")


def fetch_commons_direct(sci: str, limit: int = 15) -> list[dict]:
    import requests
    r = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": sci,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|dimensions",
            "iiurlwidth": 1280,
            "format": "json",
        },
        headers={"User-Agent": "AnimlVision/1.0 (https://animl.org; contact@animl.org)"},
        timeout=20,
    ).json()
    out = []
    for pid, p in r.get("query", {}).get("pages", {}).items():
        if "imageinfo" not in p:
            continue
        t = p["title"].lower()
        if any(k in t for k in ("map", "repro", "embryo", "webm", "round_stingray", ".png", "touch_pool")):
            continue
        ii = p["imageinfo"][0]
        u = ii.get("thumburl") or ii["url"]
        out.append({"photo_id": f"commons_{pid}", "original_url": u, "medium_url": u, "title": p["title"]})
    return out


def find_clean_bat_ray_candidate() -> tuple[Image.Image, Image.Image, dict]:
    """Fetch Bat Ray (`Myliobatis californica`, taxon_id=51491) candidates and vet `[Original Photo | Cutout]`
    with Gemma3:27b to pick a clean live swimming Bat Ray (no humans, no dead beach specimens, no murky green)."""
    import base64, requests
    from io import BytesIO
    from test_fresh_20_e2e import OLLAMA_URL, unload_ollama_model

    taxon_id = 51491
    sci = "Myliobatis californica"
    cands = []
    seen = set()
    # Check Wikimedia Commons File namespace (gsrnamespace=6) first
    for c in fetch_commons_direct(sci, limit=18):
        if c["photo_id"] not in seen:
            seen.add(c["photo_id"])
            cands.append(c)
    for c in fetch_inat_curated_taxon(taxon_id, limit=12):
        if c["photo_id"] not in seen:
            seen.add(c["photo_id"])
            cands.append(c)

    # Keep the front-gliding Bat Ray copy for comparison
    if (ARTIFACT_DIR / "fresh20_v2_20_bat_ray.png").exists():
        shutil.copyfile(ARTIFACT_DIR / "fresh20_v2_20_bat_ray.png", ARTIFACT_DIR / "fresh20_v2_20_bat_ray_front.png")

    best_pack = None
    best_score = -999.0
    for c in cands[:20]:
        if c["photo_id"] in (10226592, 261520825, 295200227, 181020313, 718373027, "commons_518338"):
            continue
        im_orig = download_pil(c["original_url"], timeout=20) or download_pil(c["medium_url"], timeout=15)
        if im_orig is None:
            continue
        norm_cut, pad_photo, stats = make_normalized_cutout(im_orig, subtype="shark_ray")
        if not stats.get("valid") or norm_cut is None:
            continue

        # Build side-by-side [Original Photo | Cutout] for Gemma3:27b vetter
        sbs = Image.new("RGB", (1024, 512), (255, 255, 255))
        sbs.paste(pad_photo.resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
        sbs.paste(norm_cut.resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
        buf = BytesIO()
        sbs.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        vlm_prompt = (
            "Left image is the original photo; Right image is the segmented cutout of a Bat Ray (Myliobatis californica).\n"
            "Inspect BOTH the Left original background and the Right cutout carefully. Return STRICT JSON:\n"
            "{\"is_underwater_swimming\": boolean (true ONLY if the Bat Ray is alive and swimming/gliding in clear water; FALSE if lying on beach sand, buried in sand, or held by a person),\n"
            " \"shows_dorsal_back_and_both_wings\": boolean (true ONLY if viewed from above or 3/4 dorsal angle showing the flat diamond back and both triangular wings; FALSE if viewed straight head-on from the front where the back/tail are hidden),\n"
            " \"cutout_has_solid_intact_wings\": boolean (true ONLY if the Right cutout has two solid, intact triangular wings with NO holes cut out of the wing and NO background rocks fused to the wing),\n"
            " \"water_clarity_and_pose_score\": number 0 to 10 (10 = crystal clear full-body swimming bat ray viewed from above/3/4 showing smooth dark dorsal back and both wings open; <=4 if murky green, front-on only, or hole in wing)}"
        )
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": "gemma3:27b",
                    "stream": False,
                    "keep_alive": "2m",
                    "format": "json",
                    "messages": [{"role": "user", "content": vlm_prompt, "images": [b64]}],
                },
                timeout=60,
            )
            res_j = json.loads(r.json()["message"]["content"])
        except Exception:
            continue

        if (
            not res_j.get("is_underwater_swimming", False)
            or not res_j.get("shows_dorsal_back_and_both_wings", False)
            or not res_j.get("cutout_has_solid_intact_wings", False)
        ):
            print(
                f"  [Bat Ray cand id={c['photo_id']} {c.get('title','')}] REJECTED (underwater={res_j.get('is_underwater_swimming')}, dorsal={res_j.get('shows_dorsal_back_and_both_wings')}, intact_wings={res_j.get('cutout_has_solid_intact_wings')})",
                flush=True,
            )
            continue
        pose_s = float(res_j.get("water_clarity_and_pose_score", 0))
        print(f"  [Bat Ray cand id={c['photo_id']} {c.get('title','')}] underwater=True, intact_wings=True, score={pose_s:.1f}", flush=True)
        if pose_s > best_score:
            best_score = pose_s
            best_pack = (norm_cut, pad_photo, c)
            if pose_s >= 8.5:
                break

    unload_ollama_model("gemma3:27b")
    torch.cuda.empty_cache()
    if best_pack is None:
        raise RuntimeError("No clean Bat Ray candidate found")
    return best_pack


def main() -> None:
    meta = json.loads((WORK / "fresh_20_metadata.json").read_text(encoding="utf-8"))

    # 1. Select a cleaner Bat Ray photo + smooth un-dappled cutout
    print("Selecting cleaner Bat Ray candidate...", flush=True)
    ray_cut, ray_photo, ray_info = find_clean_bat_ray_candidate()
    # Smooth any subtle underwater light ripples and neutralize blue/green underwater color cast toward warm dark brown
    ray_arr = np.array(ray_cut)
    fg = np.any(ray_arr < 245, axis=-1)
    fg_eroded = cv2.erode(fg.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)) > 0
    lab = cv2.cvtColor(ray_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    if lab[fg, 2].mean() < 134.0:
        lab[fg, 1] = np.clip(lab[fg, 1] + (131.0 - lab[fg, 1].mean()) * 0.9, 0, 255)
        lab[fg, 2] = np.clip(lab[fg, 2] + (138.0 - lab[fg, 2].mean()) * 0.95, 0, 255)
        ray_arr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    ray_smooth = cv2.bilateralFilter(ray_arr, d=15, sigmaColor=55, sigmaSpace=55)
    ray_Smooth2 = cv2.bilateralFilter(ray_smooth, d=15, sigmaColor=55, sigmaSpace=55)
    ray_arr[fg_eroded] = ray_Smooth2[fg_eroded]
    ray_cut = Image.fromarray(ray_arr)

    ray_photo.save(V2_WORK / "bat_ray_new_photo.png")
    ray_photo.save(WORK / "photos_selected" / "20_bat_ray.png")
    ray_cut.save(V2_WORK / "cutouts_v2" / "20_bat_ray.png")
    ray_canny = build_canny_from_repaired(ray_cut)
    ray_canny.save(V2_WORK / "canny_v2" / "20_bat_ray.png")

    # 2. Load SDXL + Canny ControlNet + animl_style_v6
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

    # 3. 19_leopard_shark is already fixed and verified at fresh20_v2_19_leopard_shark.png
    shark_out = V2_WORK / "clean_out" / "19_leopard_shark.png"
    if not (ARTIFACT_DIR / "fresh20_v2_19_leopard_shark.png").exists():
        shark_cut = Image.open(WORK / "cutouts_1024" / "19_leopard_shark.png").convert("RGB")
        shark_canny = Image.open(WORK / "canny" / "19_leopard_shark.png").convert("RGB")
        shark_depth = Image.open(WORK / "depth" / "19_leopard_shark.png").convert("RGB")
        shark_traits = meta["19_leopard_shark"]["visual_traits"]
        shark_prompt = (
            f"animl_gouache_style, a Leopard Shark, full body wildlife illustration, "
            f"{shark_traits}, {STYLE_V6}"
        )
        shark_neg = "tiger stripes, vertical stripes, zebra stripes, " + NEG_V6_UNIFIED
        pos_e, neg_e, pos_p, neg_p = encode_prompt_sdxl_multichunk(pipe, shark_prompt, shark_neg, max_chunks=3)
        gen = torch.Generator(device="cuda").manual_seed(42)
        shark_res = pipe(
            prompt_embeds=pos_e,
            negative_prompt_embeds=neg_e,
            pooled_prompt_embeds=pos_p,
            negative_pooled_prompt_embeds=neg_p,
            image=shark_cut,
            control_image=[shark_canny, shark_depth],
            strength=0.55,
            controlnet_conditioning_scale=[0.48, 0.25],
            control_guidance_start=0.0,
            control_guidance_end=[0.78, 0.60],
            guidance_scale=6.0,
            num_inference_steps=36,
            generator=gen,
        ).images[0]
        if shark_out.exists():
            shark_out.unlink()
        clean_post(shark_res, shark_out)
        shutil.copyfile(shark_out, ARTIFACT_DIR / "fresh20_v2_19_leopard_shark.png")

    # 4. Run 20_bat_ray with smooth unspotted dark-brown dorsal skin (and sanitize 'Bat' + 'eagle' so CLIP adds neither mammal bat ears nor Spotted Eagle Ray dots!)
    ray_traits = (
        "Smooth uniform dark-brown to olive-brown dorsal surface without any spots or markings, "
        "white ventral underside, broad triangular wing-like pectoral fins, blunt rounded snout, long whip-like tail"
    )
    ray_prompt = (
        f"animl_gouache_style, a smooth-skinned Myliobatis ray (Myliobatis californica), full body wildlife illustration, "
        f"{ray_traits}, {STYLE_V6}"
    )
    ray_neg = "bat ears, mammal ears, pointed ears, spots, polka dots, leopard spots, blotches, speckles, rings, patterns, " + NEG_V6_UNIFIED
    pos_e, neg_e, pos_p, neg_p = encode_prompt_sdxl_multichunk(pipe, ray_prompt, ray_neg, max_chunks=3)
    gen = torch.Generator(device="cuda").manual_seed(42)
    ray_res = pipe(
        prompt_embeds=pos_e,
        negative_prompt_embeds=neg_e,
        pooled_prompt_embeds=pos_p,
        negative_pooled_prompt_embeds=neg_p,
        image=ray_cut,
        control_image=[ray_canny, ray_canny],
        strength=0.54,
        controlnet_conditioning_scale=[0.44, 0.0],
        control_guidance_start=0.0,
        control_guidance_end=[0.74, 1.0],
        guidance_scale=6.0,
        num_inference_steps=36,
        generator=gen,
    ).images[0]
    ray_out = V2_WORK / "clean_out" / "20_bat_ray.png"
    if ray_out.exists():
        ray_out.unlink()
    clean_post(ray_res, ray_out)
    shutil.copyfile(ray_out, ARTIFACT_DIR / "fresh20_v2_20_bat_ray.png")
    print("Saved fixed 19_leopard_shark and 20_bat_ray!", flush=True)


if __name__ == "__main__":
    main()
