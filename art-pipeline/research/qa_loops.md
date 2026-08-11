# Closed-Loop QA / Auto-Retry for AI Image Pipelines — Survey (Aug 2026)

Context: automated wildlife-illustration factory (local SDXL + style LoRA + img2img, RTX 5080,
possibly hybrid with API generators). A grader flags defects (missing limbs, wrong anatomy,
glossy style drift, artifacts) and the pipeline must re-prompt / repair / regenerate without a
human in the loop. Practical lens: what a solo dev can wire into a Python pipeline this month.
Existing assets in this repo that matter: `art-pipeline/gemma_judge.py` (Ollama gemma3:12b judge
calibrated against the user's own verdicts), `generate_batch.py`, `grid_search.py`,
`photo_funnel.py`.

---

## 1. Loop frameworks: what exists

### 1.1 The pattern that actually has adoption: a plain code loop

The honest headline: there is no widely-adopted off-the-shelf "self-correcting image gen"
framework. The pattern with real production adoption is the **evaluator–optimizer loop written
in ordinary code**: generate → VLM judge → structured verdict → adjust → regenerate, with a
max-iteration cap. Everything below is either a building block for that loop or a research
prototype of it.

- **DSPy official tutorial — "Image Generation Prompt Iteration"**
  (https://dspy.ai/tutorials/image_generation_prompting/). Exactly this loop in ~30 lines:
  Flux via fal.ai generates, a VLM signature returns `{feedback, matches_desired: bool,
  revised_prompt}`, loop caps at 5 iterations. Trivially adaptable to local SDXL + Ollama.
  DSPy's value-add beyond the loop is **GEPA** (https://github.com/gepa-ai/gepa,
  https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/) — reflective prompt evolution that could
  offline-optimize the judge prompt or the per-species prompt template against your recorded
  accept/reject data. DSPy signatures support `dspy.Image` fields natively.

- **LangGraph evaluator–optimizer pattern**
  (https://medium.com/@okanyenigun/built-with-langgraph-34-evaluator-optimizer-pattern-b5e02d611c5c,
  https://learnopencv.com/langgraph-self-correcting-agent-code-generation/). Generate node →
  evaluate node → conditional edge back to generate with feedback, revision limit to bound the
  loop. Fine, but for a single linear pipeline it adds a dependency without adding much over a
  `for` loop; worth it only if you later want branching state machines (escalation tiers,
  parallel species batches with shared state).

### 1.2 ComfyUI: loops and judges inside the graph

ComfyUI can express retry loops natively since the execution-model-inversion merge
(https://github.com/comfyanonymous/ComfyUI/pull/2666), but it is awkward and the ecosystem
mostly treats it as experimental:

- **Loop/branch primitives**: `execution-inversion-demo` nodes
  (https://github.com/BadCafeCode/execution-inversion-demo-comfyui), Easy-Use ForLoopStart/End
  (example: https://github.com/o-l-l-i/ComfyUI-Olm-Easy-Use-for-Loop-Example),
  Impact Pack `ImpactConditionalBranch`
  (https://www.runcomfy.com/comfyui-nodes/ComfyUI-Impact-Pack/ImpactConditionalBranch).
- **VLM-judge nodes**: `ComfyUI_VLM_nodes` (https://github.com/gokayfem/ComfyUI_VLM_nodes,
  structured-output node included), `comfyui-ollama` (https://github.com/stavsap/comfyui-ollama)
  which talks to the same Ollama server `gemma_judge.py` already uses, and
  `ComfyUI-IF_AI_tools` (https://github.com/if-ai/ComfyUI-IF_AI_tools).
- **Auto-retry detailer**: Impact Pack ships `DetailerForEachAutoRetry`
  (https://comfyai.run/documentation/DetailerForEachAutoRetry) — detect→inpaint with retry —
  but its retry predicate is not a semantic judge.

**Verdict**: don't put the control loop in the graph. Keep control flow in Python and use
ComfyUI (if at all) as a stateless render backend via its HTTP/WebSocket API, or transpile
workflows to Python with **ComfyScript** (https://github.com/Chaoses-Ib/ComfyScript) or
**ComfyUI-to-Python-Extension** (https://github.com/pydn/ComfyUI-to-Python-Extension). Since
this pipeline already drives diffusers/sd-scripts directly, that argument resolves to: keep
doing what `generate_batch.py` does, add the loop around it.

### 1.3 Academic self-correcting T2I with usable code

- **SLD — Self-correcting LLM-controlled Diffusion** (CVPR 2024,
  https://github.com/tsunghan-wu/SLD, https://arxiv.org/abs/2311.16090). Loop = OWLv2
  open-vocab detector finds prompt/image mismatches → GPT-4 plans add/move/remove operations →
  DiffEdit-style latent ops execute them. Training-free, works on API generators too. Reality:
  ~190 stars, effectively unmaintained since early 2024, SD1.4-era base model, "results vary
  with tuning". **Steal the architecture (detector→planner→latent-edit), not the code.**
- **Idea2Img** (ECCV 2024, https://github.com/zyang-ur/idea2img,
  https://arxiv.org/abs/2310.08541). GPT-4V cyclically drafts prompts, scores draft images,
  keeps a memory of what the T2I model responds to, revises. This "feedback memory per model"
  idea is directly reusable: keep a per-species / per-defect notebook of which prompt tweaks
  fixed which judge complaints.
- **GenArtist** (NeurIPS 2024 spotlight, https://github.com/zhenyuw16/GenArtist,
  https://arxiv.org/abs/2407.05600). MLLM agent decomposes the task, picks tools from a library
  (generation, editing, verification), builds a plan tree with step-by-step verification.
  Research-grade glue; the tool-library + verify-each-step shape is the generalization of the
  tiered-repair ladder recommended in §5.
- **ReflectionFlow** (ICCV 2025, https://github.com/Diffusion-CoT/ReflectionFlow). The
  strongest academic result: FLUX fine-tuned as a "corrector" that takes (flawed image +
  textual reflection) → improved image, plus the 1M-triplet **GenRef** dataset and verifier
  models. GenEval 0.67 → 0.91 with full scaling. But training used 8×A100-80GB and inference
  is not documented for consumer cards — **not practical on a 5080 this month**, though GenRef
  is an interesting source of defect-reflection language for judge prompts.
- Surveys for orientation: test-time scaling for multimodal generation
  (https://arxiv.org/pdf/2606.08231), RL/reward alignment of diffusion models
  (https://arxiv.org/pdf/2505.17352), agentic visual generation roadmap
  (https://github.com/EvolvingLMMs-Lab/Evolving-Visual-Generation).

---

## 2. Targeted repair: cheaper than full regeneration

Maturity here is high — every piece is scriptable today, no human in the loop.

### 2.1 Detect defect region → mask → inpaint

- **Judge-grounded boxes**: modern VLMs return bounding boxes as JSON. Qwen2.5-VL (and
  successors) are explicitly trained for box/point grounding with stable JSON output
  (https://qwenlm.github.io/blog/qwen2.5-vl/, https://arxiv.org/pdf/2502.13923). The cleanest
  design: the *judge itself* emits `{defect_type, bbox, severity}` so the repair stage needs no
  second model. Gemma 3 grounding is weaker; if gemma3:12b boxes prove unreliable, run
  Qwen2.5-VL-7B in Ollama for the localization call only.
- **Text→mask fallback**: Grounded-SAM (GroundingDINO detects from text, SAM segments)
  is the standard fully-automated masking combo
  (https://blog.segmind.com/grounded-sam/,
  https://blog.segmind.com/how-to-create-better-masks-for-stable-diffusion-inpainting/). For
  animal parts ("left hind leg", "tail") GroundingDINO's open-vocab detection is a good match.
- **Detector→detailer in one pack**: ComfyUI **Impact Pack** SEGS flow (detect → crop →
  upscale → inpaint → composite back) with ultralytics YOLO detector models
  (https://github.com/ltdrdata/ComfyUI-Impact-Pack). **ADetailer**
  (https://github.com/Bing-su/adetailer) is the same pattern for A1111: YOLO/MediaPipe detect →
  mask → inpaint at high local resolution → composite; batch-capable. There is no pretrained
  "malformed animal limb" YOLO — but accumulated grader verdicts with boxes become training
  data for a custom one (see §3.4).

### 2.2 Inpainting quality without a dedicated inpaint model

- **Differential diffusion** (per-pixel denoise strength via soft masks,
  https://differential-diffusion.github.io/, https://arxiv.org/pdf/2306.00950) is built into
  core ComfyUI as a single node and works with your existing SDXL+LoRA checkpoint — no
  specialized inpaint model, no seams. Recipe: blurred/gradient mask + DifferentialDiffusion +
  InpaintModelConditioning, denoise 0.6–0.8
  (https://medium.com/code-canvas/how-to-use-differential-diffusion-for-better-inpainting-in-comfyui-bbf710a80f98).
  Also available in diffusers-land as a community technique. **This is the key trick for style
  coherence: the repair pass uses the same style LoRA, so the patch matches the house style.**
- **Fooocus inpaint patch + pre-fill**: `Acly/comfyui-inpaint-nodes` brings the Fooocus SDXL
  inpaint head plus LaMa/MAT pre-fill to any SDXL checkpoint
  (https://github.com/Acly/comfyui-inpaint-nodes) — best current option when a region must be
  *removed* (extra limb, artifact blob) rather than redrawn.
- **BrushNet / PowerPaint** plug-and-play inpaint branches
  (https://github.com/nullquant/ComfyUI-BrushNet): strong papers, but benchmark comparisons are
  mixed vs plain SDXL-inpaint (FID 13.3 for SDXL-inpaint vs 31.9 BrushNet / 19.7 PowerPaint in
  one patch-consistency study, https://arxiv.org/pdf/2510.13419) and PowerPaint V2 shows
  distortion in some tests. Try differential diffusion + Fooocus patch first.

### 2.3 img2img refinement passes

For **global** defects (glossy style drift, oversaturation) region masks are the wrong tool —
a full-frame img2img re-pass with the style LoRA at denoise 0.3–0.5 preserves composition and
re-imposes the matte look. This is the cheapest repair tier: one extra sampling pass, no
detection needed, driven purely by a "style_drift" verdict from the judge. The pipeline already
does img2img from photo references, so it is a parameter change, not new code.

### 2.4 Cost ordering (per attempt, roughly)

img2img re-pass (≈1 gen) < regional inpaint (≈0.3–1 gen + detection) < full regen with new
seed (1 gen, but loses everything good) < full regen with revised prompt (1 gen + LLM call,
highest variance). A repair-first ladder typically converts 30–70% of near-misses without
burning full regens — and near-misses (one bad leg on an otherwise-accepted illustration) are
exactly the failure mode of a locked SDXL recipe.

---

## 3. Fine-tuning from accumulated feedback

### 3.1 Diffusion-DPO (pairwise preferences)

Paper: Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization".
Tooling: **diffusers ships SD1.5 and SDXL LoRA DPO training scripts**
(https://github.com/huggingface/diffusers/blob/main/examples/research_projects/diffusion_dpo/README.md,
`train_diffusion_dpo_sdxl.py`) — flagged "highly experimental, tested in low-data regimes",
which honestly matches a solo-dev dataset anyway. SDXL DPO LoRAs demonstrably work
(https://huggingface.co/radames/sdxl-DPO-LoRA). Data requirement: *pairs* (preferred, rejected)
for the same prompt — the factory's best-of-N stage produces these for free: judge-ranked
winner vs loser from the same species prompt.

### 3.2 Diffusion-KTO (binary per-image feedback) — better fit

**Diffusion-KTO** (https://github.com/jacklishufan/diffusion-kto,
https://arxiv.org/abs/2404.04465, NeurIPS 2024) optimizes from *unpaired binary* signals —
exactly the shape of accumulated accept/reject verdicts — no reward model, no pairing, and it
reports beating SFT and Diffusion-DPO on PickScore/ImageReward evals. This is the most direct
"verdict log → better LoRA" path. Official repo is research code; budget a weekend to adapt it
to SDXL+LoRA on 16GB (grad-checkpointing + 8-bit optimizer will be needed).

### 3.3 Retraining the style LoRA on accepted outputs (self-training)

Works once or twice, then degrades. Model-collapse literature is unambiguous: recursive
training on own outputs loses distribution tails and compounds artifacts — including
specifically the **LoRA-finetuned Stable Diffusion self-consuming chain** setting
(https://arxiv.org/html/2407.17493v3; general results https://arxiv.org/pdf/2404.02305,
https://arxiv.org/pdf/2505.21677). Safe recipe if used at all:
- keep the original human-curated masters as a permanent anchor set, never train on
  synthetic-only;
- cap synthetic fraction (~20–30%) and only admit images the *human* accepted, not merely
  judge-passed (judge-passed selection amplifies judge blind spots);
- one round, then re-evaluate style drift against held-out masters — do not chain rounds.
KTO/DPO (which push *away* from rejects rather than imitating accepts) age better than SFT
self-training.

### 3.4 Reward models and other uses of the verdict log

- **Off-the-shelf scorers for best-of-N reranking / regression tests**: ImageReward
  (137k expert comparisons; also ships ReFL reward-guided fine-tuning), PickScore, HPSv2/v3
  (`pip install hpsv3`, https://pypi.org/project/hpsv3/), and **VQAScore / t2v-metrics**
  (https://github.com/linzhiqiu/t2v_metrics) — one line of Python, outperforms CLIPScore
  massively on compositional checks ("does the elk have four legs?" style prompts). Caveat:
  all are tuned to photoreal/aesthetic human preference, not a matte-gouache house style — use
  them as *sanity floors* and anatomy checks, keep the calibrated Gemma judge as the style
  authority.
- **Verdict log as detector training data**: graded defects with boxes accumulate into a
  custom ultralytics YOLO "wildlife defect detector" (garbled-limb, extra-limb, glossy-patch),
  which then makes the §2 repair stage judge-free and fast (ADetailer/Impact Pack consume
  custom YOLO weights directly).
- **Judge calibration** (already in `gemma_judge.py calibrate`): keep doing it; every disagree
  case is a judge-prompt improvement candidate — optionally optimized with GEPA (§1.1).

---

## 4. Hosted platforms with generate+judge+retry

Short version: **nobody sells the full closed loop as a product**; they sell pieces.

- **Scenario** (https://docs.scenario.com/) — API for custom-model training, generation,
  editing (expand/enhance/remove-bg), and multi-step "Workflows & Apps" chaining. Strong on
  style-locked custom models; no built-in VLM judge/retry.
- **Leonardo.ai** — Alchemy v4 pipeline + Phoenix models + API
  (https://aitoolsdevpro.com/ai-tools/leonardo-ai-guide/); enhancement pipelines, not QA loops.
- **fal.ai** — Workflows chain models as a JSON graph (only `run`/`display` node types — no
  conditionals/loops in-graph, https://lobehub.com/skills/fal-ai-community-skills-fal-workflow);
  the documented production pattern is external orchestration (e.g. Inngest) doing the
  retry/judge around fal calls
  (https://www.inngest.com/blog/how-to-build-a-production-ai-image-generation-pipeline-with-fal-ai-and-inngest).
  Best API escape hatch if the 5080 becomes the bottleneck (Flux + LoRA hosting).
- **Replicate** — raw model endpoints + deployments; "no quality layer" is precisely the gap
  third parties advertise against, e.g. Runflow's guard mode "blocks bad images and
  auto-retriggers generation" (https://www.runflow.io/compare/replicate) — evidence the
  judge+retry layer is emerging as a product category, but as a hosted photoreal-oriented
  filter, not a custom-style grader. (Replicate was acquired by Cloudflare, Nov 2025.)
- **ComfyDeploy** (https://github.com/BennyKok/comfyui-deploy) and friends (BentoML comfy-pack
  https://www.bentoml.com/blog/comfy-pack-serving-comfyui-workflows-as-apis, Baseten,
  ViewComfy) — turn ComfyUI graphs into REST endpoints. Relevant only if the render backend
  moves to ComfyUI and needs cloud scale-out; the loop still lives in your Python.

Conclusion: a hosted product will not replace the local loop — the judge is the differentiated
asset (calibrated to one person's taste on one house style), and no platform hosts *that*.

---

## 5. Recommended architecture (wire-up for this month)

Python orchestrator (extend `art_factory.py` / `generate_batch.py`), no framework needed.
Escalation ladder per species, every attempt logged:

```
Tier 0  BEST-OF-N     N seeds via locked recipe -> gemma judge ranks -> top passes? done.
Tier 1  CHEAP REROLL  structured verdict -> deterministic playbook (seed bump, CFG nudge,
                      defect-specific negative prompt append) -> regen. Max 2.
Tier 2  REPAIR        - style_drift/gloss  -> full img2img re-pass, LoRA on, denoise 0.35-0.5
                      - regional defect    -> judge/Qwen2.5-VL bbox (fallback GroundingDINO+SAM)
                                             -> gaussian-feathered mask -> differential-diffusion
                                             inpaint with SAME SDXL+LoRA, denoise 0.6-0.8
                                             -> re-judge the composite. Max 2.
Tier 3  RE-PROMPT     Idea2Img-style: judge critique + per-species feedback memory -> revised
                      prompt -> full regen. Max 2, then -> human review queue.
```

Log schema per attempt: `(species, prompt, params, image_hash, verdict_json, tier, action,
outcome)`. This log is simultaneously: judge-calibration data, Diffusion-KTO training data,
defect-YOLO training data, and the Idea2Img memory.

**Highest-leverage components (in order):**
1. **Structured judge verdict with a defect taxonomy + localization** — upgrade
   `gemma_judge.py` output to `{verdict, defects: [{type, bbox, severity}], style_axes}`.
   Everything downstream (playbook reroll, masked repair, KTO labels) hangs off this one
   schema. If Gemma boxes are unreliable, one Qwen2.5-VL grounding call per flagged image.
2. **Differential-diffusion regional repair with the style LoRA** — converts near-misses
   (the dominant failure class of a locked recipe) at ~⅓ the cost of regen while guaranteeing
   style-matched patches. Pure diffusers/ComfyUI-node code, buildable in a day or two.
3. **Verdict-log flywheel → Diffusion-KTO pass on the style LoRA** once ~500+ verdicts
   accumulate — binary accept/reject is exactly KTO's input shape; avoid naive self-training
   loops (model collapse, §3.3).

Day-one quick win: Tier 0 alone (best-of-4 + judge rerank) typically buys the largest single
quality jump per engineering hour, and the pipeline is already 90% of the way there.
