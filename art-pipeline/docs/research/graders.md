# Image-Grading Models for the AnimlVision Art Pipeline

Survey date: 2026-08-11. Target hardware: Windows 11, RTX 5080 16GB (Blackwell, sm_120).
Roles: **A** = reference-photo audit (one animal, whole body, nothing in mouth, pose, angle, age, photo-vs-drawing). **B** = generated-illustration QA (missing/faded parts, extra objects, anatomy, glossy-AI vs flat-matte style, speckle, multiple animals).

---

## TL;DR

| Role | Local default | Programmatic backstop | API upgrade path |
|---|---|---|---|
| A — photo audit | **Qwen3-VL-8B-Instruct** (Q4_K_M, llama.cpp or Ollama >= 0.12.7) | **MegaDetector v6** (count + bbox margins) + **SuperAnimal-Quadruped** keypoints (limb completeness) | **Gemini 3 Flash-Lite** (~$0.22/1k imgs) -> **Claude Haiku 4.5** (~$2.6/1k) for hard cases |
| B — output QA | **Qwen3-VL-8B** with ref+gen 2-image prompt (multi-image genuinely works, unlike Gemma 3) | **CSD/DINOv2 style probe** (glossy-vs-matte), **PAL4VST** (artifact masks), **ImageReward/HPSv2** (gross-failure floor), keypoints re-run on illustration | **Claude Haiku 4.5** batch API (~$1.9/1k with 2 images, 50% batch discount) |

Golden rule before trusting any of this: hand-label ~100–200 images per role as an eval set and measure each grader against it. Every accuracy number below is a reputation-based estimate, not a guarantee for stylized wildlife art.

---

## 1. Open VLMs that fit 16GB

### Quick comparison

| Model | Params / VRAM @ Q4 | Fine-grained & spatial reputation | Multi-image (real, not token-level) | Windows runtime | Verdict for this pipeline |
|---|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** | 8B / ~6GB + mmproj | Best-in-class small VLM: 2D/3D grounding, counting, GUI-level spatial precision | **Yes — explicitly trained on multi-image referring grounding, visual correspondence; leads BLINK/MuirBench for its size** | llama.cpp (official Qwen GGUFs), Ollama >= 0.12.7, vLLM (WSL2) | **Winner for both roles** |
| Qwen3-VL-4B | 4B / ~3.5GB | Slightly below 8B, still strong | Yes | Same | Use if VRAM must be shared with the diffusion model |
| Qwen3-VL-30B-A3B (MoE) | 30B total, 3B active / ~18GB Q4 | Above 8B | Yes | llama.cpp with `--n-cpu-moe` (experts on CPU, fast since 3B active) | Optional accuracy bump; doesn't fully fit 16GB, but CPU-offloaded MoE is usable |
| Qwen2.5-VL-7B | 7B / ~5.5GB | Was the fine-grained/grounding leader of 2024–25; superseded by Qwen3-VL | Partial (works, weaker than Qwen3-VL) | llama.cpp, Ollama | Only if you're already wired for it; otherwise skip to Qwen3-VL |
| **MiniCPM-V 4.5** (Qwen3-8B + SigLIP2) | 8B / ~5.5GB | Very strong OCR/doc; good general perception; less grounding-focused than Qwen3-VL | Yes — advertised single/multi-image/video; works in llama.cpp | llama.cpp, official Ollama (`minicpm-v4.5`), int4/GGUF/AWQ | Solid #2; worth A/B-testing against Qwen3-VL on your eval set |
| MiniCPM-V 4.6 | 1.3B edge model (Qwen3.5-0.8B base) | Efficiency play, not accuracy play | Yes | Ollama (`minicpm-v4.6`), llama.cpp | Too small for nuanced judgment; ignore |
| **InternVL3.5-8B** | 8B / ~6GB | Excellent perception benchmarks; strong fine-grained VQA | Model supports it; llama.cpp path less battle-tested | llama.cpp via bartowski GGUFs; no official Ollama entry | Good 3rd option; more friction on Windows |
| Gemma 3 12B (current baseline) | 12B / ~8GB | Decent single-image attribute QA (your experience); fixed 896px SigLIP + pan-and-scan limits fine spatial detail | **No — attends mostly to first image (your observed failure is a known pattern)** | Ollama, llama.cpp | Keep only as fallback; replace |
| Gemma 3 27B QAT | 27B / ~15GB Q4 | Better judgment, same architectural vision limits | No | Fits 16GB barely at Q4, slow, no KV headroom | Not worth it vs Qwen3-VL-8B |
| Gemma 4 (12B / E2B / E4B, 2026) | 12B / ~8GB | New; multimodal now supported in llama.cpp; multi-image behavior unproven | Unverified — test before trusting | llama.cpp, Ollama | Watch, don't adopt blind |
| Llama 3.2 Vision 11B | 11B / ~8GB | Cross-attention arch; mediocre fine-grained, no grounding, **single image only** | No | Ollama | Skip |
| Llama 4 Scout | 109B total | — | — | Doesn't fit 16GB | Skip |
| Pixtral 12B | 12B / ~8GB | Decent 2024-era; natively interleaved multi-image | Yes (native) | llama.cpp (ggml-org GGUF) | Superseded by Qwen3-VL; skip |
| Phi-4-multimodal (5.6B) | 5.6B | OCR-leaning; weak fine-grained | Limited | **No llama.cpp support**; ONNX/DirectML path | Skip |
| Molmo 7B-D (2024) | 7B | Unique strength: **pointing + counting** (PixMo data) | Limited | **No llama.cpp support**; HF transformers | Superseded by Molmo 2 |
| **Molmo 2-8B** (Dec 2025, Ai2) | 8B (Qwen3 base) / ~17GB bf16, ~9GB 4-bit | SOTA open pointing/tracking; beats old 72B Molmo; strong multi-image | Yes — multi-image + video native | HF transformers (PyTorch cu128 works natively on Windows/Blackwell); no GGUF as of this writing | **Special weapon: "point at each visible paw/wing/eye" gives you countable, checkable output for limb verification** |

Sources: [Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF), [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631), [Ollama qwen3-vl](https://ollama.com/library/qwen3-vl), [llama.cpp multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md), [MiniCPM-V repo](https://github.com/openbmb/MiniCPM-V), [Ollama minicpm-v4.5](https://ollama.com/library/minicpm-v4.5), [MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6), [InternVL3.5-8B GGUF](https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-8B-GGUF), [Pixtral GGUF](https://huggingface.co/ggml-org/pixtral-12b-GGUF), [Molmo 2 announcement](https://allenai.org/blog/molmo2), [TinyWeights 2026 local VLM roundup](https://tinyweights.dev/posts/best-local-vision-language-models-2026/).

### Why Qwen3-VL-8B specifically

- The Qwen3-VL technical report describes deliberate multi-image training ("multi-image referring grounding, visual correspondence, multi-hop reasoning") and evaluates on BLINK and MuirBench — this is exactly the capability Gemma 3 lacks and Role B needs (ref photo vs. generated output in one prompt).
- Grounding/counting/spatial precision is the family's signature (it's the model line people use for GUI agents). "Is any limb cut off by the frame edge?" and "how many animals?" are grounding questions, not chat questions.
- Official first-party GGUFs, Ollama support since 0.12.7, Apache 2.0.

### Bleeding edge caveat (Aug 2026)

Qwen3.5/3.6-generation vision GGUFs currently work in llama.cpp but **not** in Ollama (the loader doesn't handle the mmproj sidecar) — if you chase the newest Qwen generation, run llama-server directly ([Unsloth Qwen3.6 notes](https://unsloth.ai/docs/models/qwen3.6)).

### Speed on the RTX 5080

- 8B-class Q4: ~130–170 tok/s decode on a 5080 ([Microcenter 5080 benchmarks](https://www.microcenter.com/site/mc-news/article/benchmarking-ai-on-nvidia-5080.aspx), [LocalAIMaster 5090-vs-5080](https://localaimaster.com/blog/rtx-5090-vs-5080-local-ai)). VLMs run somewhat slower; budget ~60–100 tok/s.
- Image prefill (vision tower + projector) adds roughly 0.3–1.5s per 1024px image.
- A ~150-token JSON verdict lands around **3–6s/image single-stream → roughly 700–1,200 images/hour**. llama-server with `--parallel 2-4` overlaps prefill/decode for more throughput.
- Blackwell notes: use CUDA 12.8+ builds. llama.cpp prebuilt CUDA binaries and Ollama both support sm_120 now; PyTorch >= 2.7 cu128 wheels run natively on Windows, so HF-transformers models (Molmo 2, scorers below) don't need WSL.
- **vLLM is still not officially supported on Windows** — use WSL2 or Docker Model Runner if you want it; community native wheels (SystemPanic/vllm-windows) exist but have their own CUDA-13 constraints ([status overview](https://fazm.ai/t/vllm-windows-support-2026), [Docker Model Runner + vLLM](https://www.docker.com/blog/docker-model-runner-vllm-windows/)). For this workload llama.cpp is enough; skip vLLM.

---

## 2. Specialized non-VLM graders

### 2a. Human-preference / reward scorers (Role B floor filter)

| Model | Arch / size | What it's good for here | Notes |
|---|---|---|---|
| [ImageReward](https://arxiv.org/abs/2304.05977) | BLIP, ~0.5GB | Gross failure floor: mangled anatomy, incoherent images score low | `pip install image-reward`; ~10–50 img/s on 5080 |
| [PickScore](https://arxiv.org/abs/2305.01569) | CLIP-H | Same, preference-flavored | Fast; CLIP-level blindness to fine detail |
| [HPSv2](https://github.com/tgxs002/HPSv2) | CLIP-H fine-tune | Same | Mostly superseded by v3 |
| [HPSv3](https://github.com/MizzenAI/HPSv3) ([paper](https://huggingface.co/papers/2508.03789)) | **Qwen2-VL-7B based**, ICCV 2025 | Best open preference model; wide-spectrum training incl. modern generators | ~16GB bf16 — run 8-bit on the 5080, or on CPU offload. `pip install hpsv3`. HPSv3++ (2026) exists for the very latest ([arXiv 2606.14657](https://arxiv.org/html/2606.14657v1)) |
| [Q-Align / OneAlign](https://github.com/q-future/q-align) ([HF](https://huggingface.co/q-future/one-align)) | mPLUG-Owl2 ~8B, ICML 2024 | 1–5 scalar for both *quality* and *aesthetics*; `model.score([img], task_="quality")` | Fits 16GB fp16; the most "plug-in scalar grade" option |
| [VQAScore](https://linzhiqiu.github.io/papers/vqascore/) ([t2v_metrics](https://github.com/linzhiqiu/t2v_metrics)) | CLIP-FlanT5 | P("Yes") for a templated question — usable as a calibrated binary check ("Does this show a single complete otter?") | clip-flant5-**xl** (3B) fits 16GB; xxl (11B) wants ~40GB — skip xxl |

**Caveat that matters for you:** these are trained on *photorealistic-leaning* generator output and human taste. On a deliberately flat, matte, painterly house style they will systematically under-score good images and can even prefer the glossy AI look you're trying to reject. Use them only as a low-bar floor (reject bottom ~5–10%), never as the style judge. (Reward-model over-optimization toward gloss is documented: [ICCV 2025 reward-model critique](https://arxiv.org/html/2507.19002v1).)

### 2b. Aesthetic / technical IQA scorers

- [pyiqa / IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) — one `pip install pyiqa` gets MUSIQ, NIMA, LAION-Aes, TOPIQ, MANIQA, NIQE, BRISQUE etc. behind a uniform API. Useful members:
  - **MUSIQ / MANIQA / TOPIQ**: technical-distortion detectors — these are your best cheap **speckle/noise** detectors (score drops on high-frequency artifacts). Also trivially complemented by classical CV: high-frequency FFT energy or median-filter residual variance in flat regions catches speckle deterministically.
  - **NIMA / LAION-aesthetic** ([improved SigLIP v2.5 variant](https://github.com/discus0434/aesthetic-predictor-v2-5)): generic photo-taste priors; weak on stylized art. Low value here.
- Verdict: run MUSIQ (or TOPIQ) as a speckle tripwire with a threshold tuned on your own accept/reject history; skip the aesthetic ones.

### 2c. AI-artifact detection / localization (Role B)

- **[PAL4VST](https://github.com/owenzlz/PAL4VST)** (Perceptual Artifacts Localization, ICCV 2023) — segmentation model that outputs *masks* over artifact regions of generated images. Lightweight, practical, gives you "where" not just "whether" — pairs well with your matte/finalize step (re-generate or inpaint flagged regions). Degrades under distribution shift, so validate on your style.
- **[LEGION](https://github.com/opendatalab/LEGION)** (ICCV 2025 Highlight, [project](https://opendatalab.github.io/LEGION/)) — MLLM-based artifact detection + segmentation + textual explanation; SOTA and more robust than PAL4VST, but heavier to deploy. Consider only if PAL4VST misses too much.
- Newer interpretable-detection benchmark (Jan 2026): [arXiv 2601.19430](https://arxiv.org/pdf/2601.19430) — worth a skim for what artifact taxonomies exist.
- Note: Claude's own docs state it *cannot* reliably determine whether an image is AI-generated ([vision docs](https://platform.claude.com/docs/en/docs/build-with-claude/vision)) — "does this look AI-generated" is a bad prompt for *any* VLM; ask about concrete features (specular highlights, gradient banding, over-rendered fur detail) or use the embedding probe below.

### 2d. Style judgment: the thing VLMs fail at — use embeddings instead

This is the most reliable fix for the "Gemma can't judge style" problem, and it's nearly free:

1. **[CSD — Contrastive Style Descriptors](https://github.com/learn2phoenix/CSD)** ([paper](https://huggingface.co/papers/2404.01292)): ViT embedding trained to encode *style independent of content*. Embed ~50–200 approved house-style images once; score new outputs by cosine similarity to the centroid (or k-NN). Threshold = your style gate. This directly measures "flat matte painterly vs glossy AI look" better than any VLM prompt.
2. Alternative/ensemble: **DINOv2 or SigLIP embeddings + logistic regression** trained on your own accepted/rejected outputs (you already have vetting labels in `art-pipeline/vetting/`). A few hundred labels typically gets 90%+ on a consistent binary style gate, runs in milliseconds.

### 2e. Detection & pose models (programmatic Role A checks + Role B anatomy)

- **[MegaDetector v6 / PyTorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife)** ([docs](https://microsoft.github.io/CameraTraps/megadetector/)) — YOLOv9/RT-DETR wildlife detector (animal/person/vehicle), MIT license, >50 img/s on the 5080.
  - "Exactly one animal": count animal boxes (confidence-thresholded).
  - "Whole body visible" heuristic: reject if the animal bbox touches the frame edge (within a few px margin) — catches most cut-off limbs before a VLM ever runs.
  - Also use its crop as the VLM input to boost fine-grained accuracy.
  - Trained on camera-trap *photos*; fine for Role A, verify on illustrations before using in Role B.
- **[DeepLabCut SuperAnimal-Quadruped](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped)** ([Model Zoo docs](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html)) — HRNet-w32, 39 keypoints, zero-shot across essentially all quadrupeds (trained on Quadruped-80K incl. AP-10K + Stanford Dogs).
  - **Limb-completeness check**: run keypoints, require all four paws + tail-base above a confidence threshold and inside the frame. Low-confidence/missing paw keypoints = cut-off or faded limb. Works on the *photo* (Role A) and — with a validation pass — on the *illustration* (Role B missing/faded parts), since your style preserves shape.
- **[ViTPose / ViTPose++ on AP-10K](https://github.com/ViTAE-Transformer/ViTPose)** + [MMPose animal configs](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html) — 17-keypoint alternative, 50+ species. MMPose install on native Windows is mmcv-compilation pain; DeepLabCut 3.x (pure PyTorch) is the smoother Windows path. Prefer SuperAnimal unless you need ViTPose's accuracy ceiling.
- Coverage gap: quadrupeds only. **Birds/fish/herps have no good zero-shot keypoint model**; options are [X-Pose/UniPose](https://github.com/IDEA-Research/X-Pose) (open-vocabulary keypoints via text prompts, ECCV 2024 — promising but research-grade) or falling back to the VLM + Molmo-2 pointing for those classes.
- Optional: **Grounding DINO / OWLv2** open-vocabulary detection for "object in mouth/paws" ("fish in beak", "prey in mouth") as a targeted detector if the VLM proves unreliable on this attribute.

---

## 3. Cheap API options

Pricing verified Aug 2026. Assumptions per image-graded: 1 image ~1024×1024, ~500-token text prompt, ~150-token JSON output.

| Model | $/M in / $/M out | Image tokens (1024²) | ~$ per 1,000 images | Notes |
|---|---|---|---|---|
| **Gemini 2.5 Flash-Lite** | 0.10 / 0.40 | 258/tile of 768² (pre-resize to <=768 → 258 total) | **~$0.14** | **Shutdown Oct 16, 2026** — don't build new on it |
| **Gemini 3 Flash-Lite** | 0.10 / 0.40 | ~1,120 flat | **~$0.22** | The long-term bottom-dollar option |
| Gemini 2.5 Flash | 0.30 / 2.50 | ~1,032 (4 tiles) | ~$0.84 | |
| Gemini 3 Flash | 0.50 / 3.00 | ~1,120 | ~$1.26 | Noticeably smarter tier |
| **GPT-5-mini** | 0.25 / 2.00 | patches × 1.62 ≈ ~1,660 | ~$0.85 | Patch budget capped at 1,536 patches |
| GPT-4.1-mini | 0.40 / 1.60 | ~1,660 | ~$1.10 | Superseded by 5-mini |
| **Claude Haiku 4.5** | 1.00 / 5.00 | ceil(w/28)×ceil(h/28) = 1,296 | **~$2.55** | Best judgment per dollar of the three; multi-image well-documented |
| Claude Sonnet (spot checks) | 3.00 / 15.00 | 1,296 | ~$7.7 | Only for auditing the auditors |

- All three vendors offer **~50% batch-API discounts** — overnight QA runs at half the above (Haiku ~$1.3/1k, Gemini 3 Flash-Lite ~$0.11/1k).
- Role B with ref+gen (2 images) roughly doubles the image-token line: Haiku ~$3.9/1k, Gemini 3 Flash-Lite ~$0.33/1k, GPT-5-mini ~$1.3/1k (undiscounted).
- At your scale (~1,500 species × a handful of refs + a few generations each ≈ 10–50k gradings), even Haiku end-to-end is $30–150 — API cost is *not* the constraint; iteration speed and rate limits are.
- Token-accounting sources: [Claude vision docs](https://platform.claude.com/docs/en/docs/build-with-claude/vision) (28×28-pixel patches; 1000×1000 = 1,296 tokens ≈ $1.30/1k images on Haiku — their own example), [Gemini token docs](https://ai.google.dev/gemini-api/docs/tokens) (258 tokens per 768² tile; <=384px = 258 flat) and [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [OpenAI images & vision guide](https://developers.openai.com/api/docs/guides/images-vision) (patch-based, 1,536-patch cap, 1.62× multiplier for the mini tier; see also [community clarification](https://community.openai.com/t/gpt-5-mini-image-input-token-calculation-discrepancy-with-official-faq-formula/1344040)), [Haiku 4.5 pricing](https://openrouter.ai/anthropic/claude-haiku-4.5), [Gemini shutdown/pricing tracker](https://tokencostcalculators.com/blog/gemini-3-api-pricing-2026/), [GPT-5-mini](https://developers.openai.com/api/docs/models/gpt-5-mini).
- Practical pick: **Gemini Flash-Lite tier for bulk binary attributes** (cheap enough to triple-vote), **Haiku 4.5 for the nuanced calls** (age class, subtle anatomy, final accept/reject) and as the disagreement-breaker.

---

## 4. Recommendations

### Role A — reference-photo audit

Pipeline order (cheap → expensive):
1. **MegaDetector v6**: animal count != 1 → reject; bbox touching frame edge → reject/flag (cut-off body). Also produces the crop.
2. **SuperAnimal-Quadruped** (quadrupeds only): all-paws-confident check → whole-body confirmation, plus pose skeleton you can classify geometrically (standing vs lying via leg-joint angles).
3. **Qwen3-VL-8B Q4** (llama.cpp `llama-server` + JSON schema): one structured call per photo for the remaining attributes — nothing in mouth, pose label, viewing angle, adult/juvenile, photo-vs-drawing. Feed the MegaDetector crop, not the full frame.
4. **API escalation**: attributes where local confidence is low (juvenile/adult is the known weak one — plumage/pelage cues are species-specific) → Haiku 4.5 or Gemini 3 Flash.

Expected accuracy (single binary attribute, decent photos): one-animal / whole-body ~95%+ with the detector ensemble; nothing-in-mouth ~90%; pose/angle ~85–95%; photo-vs-drawing ~99%; **adult-vs-juvenile 70–85% — treat as advisory, keep it in the human vetting UI**.

### Role B — generated-output QA

1. **Deterministic layer**: speckle via MUSIQ/TOPIQ (pyiqa) + FFT high-frequency check; **style gate via CSD cosine similarity** to approved-set centroid (this is the glossy-vs-matte judge — do not ask a VLM); ImageReward/Q-Align floor to kill catastrophes.
2. **Structural layer**: MegaDetector (multiple animals / extra objects that look animal-like) + SuperAnimal keypoints on the illustration (missing/faded limbs) — validate once on ~50 illustrations; your flat style preserves silhouettes so photo-trained models usually transfer, but confirm.
3. **VLM layer**: Qwen3-VL-8B two-image prompt — "Image 1 is the reference photo, Image 2 the illustration. Report: missing or extra body parts, extra objects, anatomy errors vs reference, number of animals" with JSON schema. This works because Qwen3-VL actually attends across images (Gemma 3 does not).
4. **Optional PAL4VST** masks to drive targeted inpainting instead of full re-rolls.
5. **API final gate** (batch, overnight): Haiku 4.5 ref+gen comparison on everything that passed local checks, ~$2/1k batched. Disagreement with local verdict → human queue.

Expected accuracy: extra objects / multiple animals ~95% (detector + VLM agree); missing-limb catch rate ~85–95% with the keypoint ensemble (VLMs alone are notoriously weak at *absence* detection — never rely on the VLM alone here); style gate 90%+ once the CSD threshold is tuned on your labels; "wrong anatomy" beyond missing parts is the hardest (~70–85%) — this is where the API escalation earns its keep.

### Integration notes (Windows / 5080 specifics)

- **llama.cpp serving**: `llama-server -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf --mmproj mmproj-...-F16.gguf -ngl 99 -c 8192 --parallel 2` → OpenAI-compatible `/v1/chat/completions`; pass multiple images as multiple `image_url` content parts; enforce JSON with `response_format`/GBNF grammar. Ollama alternative: `ollama run qwen3-vl:8b` (needs >= 0.12.7) with `format: json` — but llama.cpp gives you grammar-constrained output and better multi-image control.
- **VRAM contention**: the diffusion generator and an 8B VLM don't co-reside comfortably in 16GB. Phase the pipeline (generate batch → free → grade batch), or keep only the small always-on graders resident (CSD/DINOv2/MegaDetector/pyiqa total < 2GB) and Qwen3-VL-4B (~3.5GB) if you want a resident VLM during generation.
- **Python stack**: everything in sections 2 runs on native Windows with PyTorch cu128 (Blackwell OK). DeepLabCut 3.x for SuperAnimal (avoid MMPose/mmcv on native Windows). Molmo 2 via transformers with 4-bit bitsandbytes if you adopt pointing-based checks.
- **Prompting discipline that measurably helps**: one attribute per question (or a strict JSON schema with per-field enums), image before text, ask for per-field confidence, and majority-vote 3 samples at temp ~0.4 for borderline attributes. For Role B, always name the images ("Image 1: reference photo… Image 2: illustration…") — this matters for every multi-image model, API or local.
- **Calibration loop**: log every grader verdict vs. your human vetting decisions (`vetting/candidates.json` already has the labels); recompute per-attribute precision/recall monthly; route only low-precision attributes to the API tier. This turns the API bill into a decreasing function of pipeline maturity.

### Sources index

Local VLMs: [Qwen3-VL GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) · [Qwen3-VL tech report](https://arxiv.org/abs/2511.21631) · [Ollama qwen3-vl](https://ollama.com/library/qwen3-vl) · [llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) · [MiniCPM-V](https://github.com/openbmb/MiniCPM-V) · [InternVL3.5 GGUF](https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-8B-GGUF) · [Molmo 2](https://allenai.org/blog/molmo2) · [5080 benchmarks](https://www.microcenter.com/site/mc-news/article/benchmarking-ai-on-nvidia-5080.aspx) · [vLLM-on-Windows status](https://fazm.ai/t/vllm-windows-support-2026)
Scorers: [HPSv3](https://github.com/MizzenAI/HPSv3) · [HPSv2](https://github.com/tgxs002/HPSv2) · [ImageReward](https://arxiv.org/abs/2304.05977) · [Q-Align](https://github.com/q-future/q-align) · [VQAScore](https://linzhiqiu.github.io/papers/vqascore/) · [pyiqa](https://github.com/chaofengc/IQA-PyTorch) · [aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5)
Artifacts & style: [PAL4VST](https://github.com/owenzlz/PAL4VST) · [LEGION](https://github.com/opendatalab/LEGION) · [CSD](https://github.com/learn2phoenix/CSD) · [CSD paper](https://huggingface.co/papers/2404.01292)
Detection & pose: [PyTorch-Wildlife / MegaDetector v6](https://github.com/microsoft/Pytorch-Wildlife) · [SuperAnimal-Quadruped](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped) · [DLC Model Zoo](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html) · [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) · [MMPose animal](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html) · [X-Pose](https://github.com/IDEA-Research/X-Pose)
APIs: [Claude vision & token math](https://platform.claude.com/docs/en/docs/build-with-claude/vision) · [Haiku 4.5 pricing](https://openrouter.ai/anthropic/claude-haiku-4.5) · [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) · [Gemini tokens](https://ai.google.dev/gemini-api/docs/tokens) · [OpenAI images & vision](https://developers.openai.com/api/docs/guides/images-vision) · [GPT-5-mini](https://developers.openai.com/api/docs/models/gpt-5-mini)
