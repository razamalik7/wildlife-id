# Image Generation Options for Style-Consistent Wildlife Illustrations at Scale

Research date: 2026-08-11. Target: ~1,400 US vertebrate species, ~6 candidates each (~8,400 images),
conditioned on (a) fixed house style (soft matte painterly field-guide illustration, currently a
self-trained SDXL style LoRA) and (b) a per-species cutout reference photo for anatomy/markings.
Hardware: RTX 5080 16GB, Windows. Current pipeline: SDXL + style LoRA + img2img @ denoise 0.5–0.6.

---

## 1. Local open-weight models (16GB-viable, commercial-safety assessed)

### Quick verdict table

| Model | Params | License for product art | Fits 16GB (gen) | LoRA train on 16GB | Quality vs SDXL | Verdict |
|---|---|---|---|---|---|---|
| SDXL | 2.6B | OpenRAIL++-M — fully commercial-safe | Yes, easily | Yes (mature) | baseline | Current; quality ceiling |
| SD 3.5 Medium / Large | 2.5B / 8B | Community License: free commercial **< $1M/yr revenue**, must register | Yes / tight | Yes / tight | ≈ or slightly better; weak finetune ecosystem | Skip — no upside |
| FLUX.1-dev / Krea-dev | 12B | **Non-commercial** (NCL v2.0) — paid self-serve license required | fp8 yes | Yes (tight) | Much better | License cost + friction |
| FLUX.1-schnell | 12B | Apache 2.0 | Yes | Poor target (4-step distilled) | Better but stiff | Use Chroma instead |
| Chroma1-HD (schnell derivative) | 8.9B | **Apache 2.0** | fp8 yes | Yes | FLUX-class | Legit commercial FLUX-class option |
| FLUX.2-dev | 32B | Non-commercial (NCL v2.0) | **No** (~32GB fp8, ~19GB Q4) | No | Top-tier | Out of reach |
| FLUX.2 klein 4B | 4B | **Apache 2.0** | Yes | Yes | > SDXL | Good; weaker than 9B |
| FLUX.2 klein 9B | 9B | Non-commercial (Builder license to go commercial) | fp8 yes | Yes (~14GB fp8) | Strong | Only with paid BFL license |
| PixArt-Σ | 0.6B | OpenRAIL (commercial OK) | Yes | Yes | ≤ SDXL finetunes; ecosystem dormant | Skip |
| Lumina-Image 2.0 | 2.6B | **Apache 2.0** | Yes | Yes | ≈ SDXL+, small ecosystem | Marginal |
| Sana / Sana 1.5 | 0.6–4.8B | **Weights research/eval only** (NVIDIA NSCL v2-custom) | Yes | — | Fast, mid quality | **Not commercial-safe** |
| HunyuanDiT | 1.5B | Tencent community (commercial OK, geo-excludes EU/UK/KR) | Yes | Yes | ≈ SDXL | Skip |
| HunyuanImage 3.0 | 80B MoE | Tencent community (geo restrictions) | **No** (24GB min at 4-bit) | No | Top-tier | Out of reach |
| Qwen-Image / -2512 | 20B | **Apache 2.0** | fp8 + offload (slow) | Yes (musubi/ai-toolkit 16GB profiles) | Clearly better; 2512 fixed realism/fur detail | Strong option, but heavy |
| Qwen-Image-Edit 2509/2511 | 20B | **Apache 2.0** | fp8 ~16GB | Yes | Editing/reference SOTA (open) | Key tool — see §2 |
| Z-Image Turbo / Base | 6B | **Apache 2.0** | Yes — BF16 in 16GB, FP8 in 8GB | Yes (ai-toolkit; Base is undistilled) | #1 open on Artificial Analysis arena at release; beats FLUX.2-dev, HunyuanImage 3.0, Qwen-Image | **Best fit for this project** |
| AuraFlow | 6.8B | Apache 2.0 | Yes | — | Stalled at v0.3 (2024), dead ecosystem | Skip |
| HiDream-I1 | 17B | **MIT** | nf4 ~16GB | Tight (24GB usually advised) | > FLUX-dev on HPS | Good license, heavy for 16GB |

### Details and 2026 licensing reality

**FLUX.1-dev — nothing fundamentally changed; self-serve commercial licensing added.**
FLUX.1-dev (and Fill/Kontext-dev/Krea-dev, plus FLUX.2-dev) remain under the FLUX [dev]
Non-Commercial License, now v2.0 (https://bfl.ai/legal/non-commercial-license-terms). Nuance that
matters: v2.0 says "You may use Output for any purpose (including for commercial purposes)" — but
the Non-Commercial Purpose definition explicitly **excludes** "use (a) for revenue-generating
activity." Running the model to produce art assets for a commercial game is revenue-generating use
of the *model*, so the safe read is you need a commercial license, and BFL now sells them
self-serve (https://bfl.ai/licensing, https://help.bfl.ai/articles/9272590838-self-serve-dev-license-overview-pricing):
- **Builder**: 10K imgs/mo, 1 domain, LoRA + fine-tune rights, FLUX.2 [klein] models; price shown only in dashboard (not public).
- **Platform**: 100K imgs/mo, klein 9B Base + FLUX.2-dev. **Professional/Enterprise**: contact sales.
So a legal FLUX path exists but is a recurring paid dependency for a solo project.

**FLUX.1-schnell** is Apache 2.0, but it's a 4-step guidance-distilled model: img2img denoise
control is coarse, LoRA training on it is a poor experience, and quality trails dev. The community
answer is **Chroma1-HD** (lodestones, 8.9B, Apache 2.0, de-distilled schnell retrain,
https://huggingface.co/lodestones/Chroma1-HD) — the practical "FLUX-quality with a real license"
base, trainable and active through 2026 (v3 checkpoints on Civitai).

**FLUX.2 klein** (released with FLUX.2, late 2025): the 4B is Apache 2.0; the **9B is NCL** —
commercial only via the Builder tier. Klein 9B Base LoRA training fits ~14GB at fp8
(https://www.runcomfy.com/trainer/ai-toolkit/flux-2-klein-16gb-vram-training,
https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora). FLUX.2-dev itself is 32B:
~32GB fp8, ~19GB GGUF Q4 — a 24GB-card model (https://willitrunai.com/image-models/flux-2-dev).

**SD 3.5 Medium/Large**: Stability AI Community License — free commercial use below $1M annual
revenue but you must register, and the license *terminates* if you cross $1M
(https://stability.ai/license, https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md).
Fine for AnimlVision's scale legally, but the model never built a finetuning ecosystem and offers
little quality upside over a good SDXL finetune. Skip.

**Qwen-Image** (Alibaba, Apache 2.0, 20B MMDiT): the December 2025 **Qwen-Image-2512** update
specifically improved natural detail — "notably more detailed rendering of landscapes, animal fur,
and other natural elements" (https://huggingface.co/Qwen/Qwen-Image-2512,
https://www.alibabacloud.com/blog/qwen-image-2512-finer-details-greater-realism_602772). LoRA
training on 16GB (even less) is proven via kohya's **musubi-tuner** (16GB VRAM profiles,
https://github.com/kohya-ss/musubi-tuner/discussions/735) and ai-toolkit. ControlNet union models
exist (InstantX), ComfyUI support is first-class. Cost of the 20B size on a 16GB card: fp8 with
block offloading → slower iteration than a 6B model.

**Z-Image** (Alibaba Tongyi, Apache 2.0, https://github.com/Tongyi-MAI/Z-Image):
- **Z-Image-Turbo** (Nov 2025): 6B, 8-step distilled, sub-second on datacenter GPUs, a few seconds
  on consumer cards; BF16 fits 16GB, FP8 in 8GB (https://willitrunai.com/image-models/z-image-turbo).
  Ranked #1 open-weights on Artificial Analysis Image Arena at release, above FLUX.2-dev,
  HunyuanImage 3.0 and Qwen-Image (https://www.rundiffusion.com/z-image).
- **Z-Image-Base** (Jan 27, 2026): the undistilled foundation checkpoint, released explicitly for
  fine-tuning/LoRA training; ControlNet stack exists (canny/depth/openpose/lineart)
  (https://gaga.art/blog/z-image-base/, https://www.apatero.com/blog/z-image-base-controlnet-guide).
  LoRA training works in ai-toolkit on consumer VRAM
  (https://github.com/FurkanGozukara/Stable-Diffusion/wiki/Z-Image-Turbo-LoRA-training-with-AI-Toolkit-and-Z-Image-ControlNet-Full-Tutorial-for-Highest-Quality).
- **Z-Image-Edit**: announced, still unreleased as of mid-2026
  (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/156).
Best quality-per-VRAM of anything Apache-licensed; the ecosystem (ControlNets, trainers, ComfyUI)
matured rapidly through H1 2026.

**Sana**: despite "open" branding, Sana 1.5 weights ship under NVIDIA's NSCL v2-custom terms where
"non-commercially means for research or evaluation purposes only"
(https://huggingface.co/Efficient-Large-Model/SANA1.5_4.8B_1024px,
https://github.com/NVlabs/Sana/issues/157). Not safe for product art.

**HiDream-I1**: 17B, genuinely MIT (https://github.com/HiDream-ai/HiDream-I1), strong quality;
runs in ~16GB only at nf4, LoRA training realistically wants 24GB. A fallback if you upgrade GPUs.

**HunyuanImage 3.0**: 80B MoE, 4-bit needs 24GB+ (https://willitrunai.com/image-models/hunyuan-image-3);
Tencent community license also carves out EU/UK/South Korea. Not practical locally.

---

## 2. Style-consistency tech beyond plain img2img

Ranked by fit for "trained house style + per-species anatomy photo":

1. **Trained style LoRA (keep this — it's the strongest style lock).** Nothing training-free
   matches a LoRA trained on your own curated style set for *locking* a fixed house style. The move
   is not to replace the LoRA but to retrain it on a stronger base (Z-Image-Base or Qwen-Image).

2. **ControlNet (depth + lineart/canny from the cutout photo) — the anatomy anchor.** Extracting
   depth + soft-edge/lineart from the reference photo and conditioning on those lets you raise
   img2img denoise from 0.5–0.6 to 0.75–0.95 (or drop the init image entirely). That's the single
   biggest lever: high denoise = full style adoption from the LoRA, while ControlNet preserves
   silhouette, pose, limb count, and marking placement. Z-Image-Base and Qwen both have
   canny/depth/lineart ControlNets in 2026 (https://www.apatero.com/blog/z-image-base-controlnet-guide).
   Practical recipe: depth (strength ~0.5) + lineart (~0.4), end conditioning early (~60–70% of
   steps) so the painterly style can loosen edges in the final steps.

3. **Qwen-Image-Edit 2511 (Apache 2.0) — reference-driven restyle as an alternative pipeline.**
   Takes up to 3 input images + an instruction; preserves structure/identity of the source while
   applying transformations; fp8 edits in ~16GB (https://blog.comfy.org/p/qwen-image-edit-2511-and-qwen-image,
   https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511). Feed (species photo + style
   exemplar) and instruct "repaint in this illustration style." There is even a dedicated
   style-transfer LoRA for it (https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Style-Transfer).
   This is the closest local equivalent to what Nano Banana Pro does in one shot, with a real
   commercial license. Weakness: per-image style drift vs a trained LoRA — mitigate by also
   training a style LoRA *for the Edit model* (musubi-tuner supports Qwen-Image-Edit LoRA training).

4. **IP-Adapter / InstantStyle**: mature on SDXL, mediocre ports on FLUX (XLabs). Known failure
   mode for your use case: content leakage from the style image and weakened prompt/pose control;
   research comparisons consistently show them below newer methods for style fidelity
   (https://rb-modulation.github.io/). With a trained style LoRA in hand, IP-Adapter style
   injection is redundant; IP-Adapter *content* conditioning from the animal photo is strictly
   worse than ControlNet for anatomy.

5. **StyleAligned**: shares attention across a batch for a uniform look; memory-heavy, degrades
   detail via DDIM inversion, and doesn't help per-species anatomy. Not production-suited here.

6. **RB-Modulation** (ICLR 2025, https://rb-modulation.github.io/): best training-free style
   personalization in head-to-heads vs InstantStyle/IP-Adapter/StyleAligned, no ControlNet needed —
   but research code on Stable Cascade, no ComfyUI production ecosystem. Watch, don't build on.

7. **FLUX.1 Redux**: style/variation adapter for FLUX.1-dev (non-commercial base). Good style
   transfer, but license and the same drift problem as all image-prompt methods.

8. **USO (ByteDance, CVPR 2026, https://github.com/bytedance/USO)**: unified style-ref + subject-ref
   generation — conceptually *exactly* this task (style image + subject image → styled subject) and
   available in ComfyUI (https://blog.comfy.org/p/uso-available-in-comfyui). Built on FLUX.1-dev →
   non-commercial without a BFL license. If you ever buy the Builder license, USO is worth a bake-off.

9. **Regional approaches**: unnecessary for single-subject-on-plain-background compositions; the
   existing matte/cutout step already isolates the subject.

**Recommended stack:** style LoRA (retrained) + depth/lineart ControlNet from photo + img2img init
at high denoise, with a Qwen-Image-Edit-2511 pass as (a) an alternate generator for hard species and
(b) a marking-correction step ("match the wing bars to the reference photo").

---

## 3. API services

| Service | Per-image | Per 1,000 | Style lock | Anatomy ref photo | Output ownership |
|---|---|---|---|---|---|
| Google Nano Banana Pro (gemini-3-pro-image) | $0.134 (1K/2K), $0.24 (4K); **batch $0.067 / $0.12** | $134 std / $67 batch | Per-request style refs only (no finetune) | Yes — up to 14 ref images (6 object + 5 person slots) | Yours; commercial OK; invisible SynthID watermark always |
| Google Gemini 2.5 Flash Image (Nano Banana) | $0.039; batch $0.0195 | $39 / $19.50 batch | Same (weaker adherence) | Yes (multi-image input) | Same |
| OpenAI gpt-image-1.5 | $0.009 / $0.034 / $0.133 (low/med/high 1024²); batch −50% | $34 med / $17 batch-med | None persistent; per-request image refs | Yes (edits/reference input) | Yours |
| Recraft V4 / V4.1 | $0.035–0.04 gen; img2img $0.044; custom style creation $0.044 one-time | ~$40–44 | **Custom Style from your reference set — reusable style_id** (designed for brand-consistent illustration) | Yes — image-to-image endpoint combinable with style | Full ownership on API/paid |
| Scenario | ~2–15 CU/img; 5,000 CU ≈ 500 FLUX-dev imgs on $45/mo Pro → ~$0.09 eff. | ~$90 eff. | **Custom model/LoRA training** (10–30 style imgs) on FLUX etc. | Yes — img2img + ControlNet modes | Full commercial license on paid plans |
| Leonardo.ai | ~$0.003–0.007 platform; API $9/mo + credit packs | ~$5–10 | Custom model training (10–20 imgs, paid tiers) | Yes — img2img, ControlNet | Yours on paid |
| Ideogram V4 | $0.03 / $0.06 / $0.10 (turbo/default/quality); style/character ref → $0.10–0.20 | $60–150 | Style reference images (per request) | Character reference (animal support weak) | Yours |
| BFL FLUX API | klein 4B $0.014, klein 9B $0.015, FLUX.2 pro $0.03 (MP-scaled) | $15–30 | **Hosted finetuning: fine-tuned klein/dev endpoints, billed at base rates during public beta** | Yes — multi-reference input on FLUX.2 endpoints | Yours (API terms) |

Sources: https://openrouter.ai/google/gemini-3-pro-image, https://ai.google.dev/gemini-api/docs/pricing,
https://developers.googleblog.com/introducing-gemini-2-5-flash-image/, https://pricepertoken.com/gpt-image-pricing,
https://www.aifreeapi.com/en/posts/gpt-image-1-5-pricing-api, https://www.recraft.ai/pricing?tab=api,
https://www.recraft.ai/docs/api-reference/getting-started, https://www.recraft.ai/blog/ownership-and-commercial-use-faq,
https://help.scenario.com/en/articles/api-usage-and-credits-creative-units, https://help.scenario.com/en/articles/pricing-plans/,
https://leonardo.ai/pricing, https://developer.puter.com/tutorials/ideogram-api-pricing/,
https://kie.ai/blog/ideogram-v4-pricing, https://docs.bfl.ml/quick_start/pricing, https://bfl.ai/pricing.

### Notes per service

- **Nano Banana Pro** (the current one-shot quality leader you observed): API accepts up to 14
  reference images per request — so each call can carry 2–3 house-style exemplars + the species
  photo + a text brief. No persistent style finetune exists, so style consistency rests on your
  fixed style board + prompt template; drift across 1,400 species is the real risk vs a LoRA.
  All outputs carry invisible SynthID (API outputs skip the visible watermark). Google claims no
  ownership of outputs; commercial use permitted under Gemini API terms
  (https://www.glbgpt.com/hub/can-i-use-gemini-ai-images/). Batch mode halves cost. gpt-image-1
  retires Oct 23, 2026 — anything OpenAI should target gpt-image-1.5.
- **Recraft** is the only API whose style mechanism matches your architecture: upload your curated
  style set once → persistent `style_id` → every generation locked to it; and it has image-to-image
  to carry the photo's anatomy. At ~$0.044/img it's also cheap. Caveat: img2img strength control +
  fidelity for fine species markings needs a hands-on trial; V4.1 Pro tier ($0.21) exists for
  higher quality.
- **Scenario** is a managed version of what you already do locally (train a style LoRA on FLUX,
  then img2img/ControlNet against it) — full commercial license on paid plans, ~$45–75/mo. Worth it
  only if you want to stop maintaining local training.
- **BFL hosted finetune** on FLUX.2 klein 9B is a sleeper option: train a style LoRA via their
  trainer (or upload), then generate at $0.015/img with the license question handled by API terms —
  $126 for the whole 8,400-image run.
- **Ideogram/Leonardo**: character-reference and platform-LoRA features are person/character-tuned;
  weakest fit for anatomically-exact wildlife.

---

## 4. Recommended architecture

**Hybrid: local Apache-licensed pipeline as the workhorse + Nano Banana Pro batch as the rescue/hero lane.**

### Primary (local, $0 marginal, no license risk)

1. **Base model: Z-Image** — Turbo for candidate volume, Base for training. Apache 2.0, 6B fits the
   5080 in BF16 with headroom for ControlNet + LoRA, ~2–4s/image. Retrain the house-style LoRA on
   **Z-Image-Base** with ai-toolkit (same curated set that trained the SDXL LoRA; 16GB is enough).
2. **Anatomy conditioning:** from the existing cutout photo, extract depth + lineart; condition via
   Z-Image ControlNets; keep the cutout as img2img init but raise denoise to ~0.8 now that
   structure is held by ControlNet. This is the direct fix for "SDXL base quality limits results" —
   the style LoRA finally gets enough denoise room to paint.
3. **Correction/edit pass: Qwen-Image-Edit 2511** (Apache, fp8 in 16GB) for marking fixes against
   the reference photo, and as an alternate generator for species where ControlNet misreads the
   photo (long thin limbs, splayed wings, eels). Swap in Z-Image-Edit when it ships.
4. Keep the existing ONNX matte/finalize step unchanged.
5. Fallback base if Z-Image style training disappoints: **Qwen-Image-2512** (Apache, better fur/
   nature detail, heavier) or **Chroma1-HD** (Apache, FLUX-class aesthetics).

### Secondary (API lane)

Route QC failures and hero/marketing renders to **Nano Banana Pro (batch)**: style board (2–3 locked
exemplars) + species photo + fixed prompt template. Optionally trial **Recraft custom style +
image-to-image** as a cheaper structured alternative ($0.044 vs $0.067).

### Cost estimates

Initial build — 1,400 species × 6 candidates = 8,400 images:

| Scenario | Cost |
|---|---|
| All local (Z-Image on 5080; ~3s/img ≈ 7 GPU-hours + retrain time) | **≈ $0 marginal** (electricity ~$5; style LoRA retrain ~1–2 days effort) |
| Hybrid (85% local + 15% hard species via NBP batch ≈ 1,260 imgs × $0.067) | **≈ $85** |
| All Nano Banana Pro batch / standard | $563 / $1,126 |
| All Recraft custom-style img2img | ≈ $370 |
| All gpt-image-1.5 medium batch | ≈ $143 |
| All BFL klein-9B finetuned endpoint | ≈ $126 |

Ongoing churn — assume ~25% of species regenerated per quarter (350 species × 6 = 2,100 imgs):

| Scenario | Per quarter | Per year |
|---|---|---|
| Local | ≈ $0 | ≈ $0 |
| Hybrid (15% API share) | ≈ $21 | ≈ $85 |
| All NBP batch | ≈ $141 | ≈ $563 |
| All Recraft | ≈ $92 | ≈ $370 |

At these volumes even the all-API path is cheap in dollars; the decisive factors are **style lock**
(trained LoRA > per-request reference images for 1,400-species consistency), **license cleanliness**
(Apache local stack has zero recurring terms; Gemini adds SynthID watermarks), and **iteration
control** (denoise/ControlNet knobs vs prompt-only steering). Hence: local-first, API for the tail.
