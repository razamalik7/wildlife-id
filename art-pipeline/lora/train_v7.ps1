# AnimlVision style LoRA v7 - SDXL, RTX 5080 (16GB)
# 45 Harmonized v3 Illustrations (4x 6x-anchor + 37x 4x-standard + 4x 2x-multipose = 180 samples/epoch),
# batch_size=1 + gradient_accumulation_steps=2 (effective batch_size=2, 90 steps/epoch),
# UNet (1e-4) + Text Encoder (2e-5), max_token_length 225, 2000 steps.
#
# Run AFTER the GPU is free:  powershell -File art-pipeline\lora\train_v7.ps1

param(
    [switch]$Force
)

$ErrorActionPreference = "Continue"

# Pre-flight VRAM safety check
$freeVramRaw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($freeVramRaw)) {
    Write-Error "Failed to query GPU free VRAM via nvidia-smi."
    exit 1
}
$freeVramMiB = [int](($freeVramRaw | Select-Object -First 1).Trim())
Write-Host ("RTX 5080 free VRAM: {0} MiB" -f $freeVramMiB)
if ($freeVramMiB -lt 10000 -and -not $Force) {
    Write-Warning ("Free VRAM ({0} MiB) is below 10000 MiB required for comfortable SDXL LoRA training. Close GPU-heavy processes or pass -Force to override." -f $freeVramMiB)
    exit 1
}

$env:PYTHONIOENCODING = "utf-8"
$lora = "C:\Users\nars7\Projects\aniML-vision\art-pipeline\lora"

& "$lora\venv\Scripts\python.exe" "$lora\sd-scripts\sdxl_train_network.py" `
  --pretrained_model_name_or_path "$lora\base\sd_xl_base_1.0.safetensors" `
  --dataset_config "$lora\dataset_v7.toml" `
  --output_dir "$lora\output" `
  --output_name "animl_style_v7" `
  --save_model_as safetensors `
  --network_module networks.lora `
  --network_dim 32 `
  --network_alpha 16 `
  --learning_rate 1e-4 `
  --unet_lr 1e-4 `
  --text_encoder_lr 2e-5 `
  --lr_scheduler cosine `
  --lr_warmup_steps 100 `
  --optimizer_type AdamW8bit `
  --gradient_accumulation_steps 2 `
  --max_token_length 225 `
  --max_train_steps 2000 `
  --save_every_n_steps 250 `
  --sample_every_n_steps 250 `
  --sample_prompts "$lora\sample_prompts_v7.txt" `
  --sample_sampler euler_a `
  --sdpa `
  --mixed_precision bf16 `
  --save_precision bf16 `
  --gradient_checkpointing `
  --cache_latents `
  --cache_latents_to_disk `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Exporting *_unet_only.safetensors variants for key checkpoints..."
@'
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file

out_dir = Path(sys.argv[1])
for stem in ("animl_style_v7-step00001250", "animl_style_v7-step00001500", "animl_style_v7-step00001750", "animl_style_v7"):
    src = out_dir / f"{stem}.safetensors"
    if not src.exists():
        print(f"Skipping missing {src}")
        continue
    sd = load_file(str(src))
    unet_sd = {k: v for k, v in sd.items() if k.startswith("lora_unet_")}
    dst = out_dir / f"{stem}_unet_only.safetensors"
    save_file(unet_sd, str(dst))
    print(f"Saved {dst.name} ({len(unet_sd)}/{len(sd)} tensors)")
'@ | & "$lora\venv\Scripts\python.exe" - "$lora\output"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
