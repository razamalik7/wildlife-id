# AnimlVision style LoRA v6 - SDXL, RTX 5080 (16GB)
# 26 Masters (balanced 6x/4x/2x repeats with Marten/Macaque fur-anchor boost),
# Marten-aligned flat cel-shaded + subtle warm brown-umber ink fur STYLE_CLAUSE,
# UNet (1e-4) + Text Encoder (2e-5), max_token_length 225, 1500 steps.
#
# Run AFTER the GPU is free:  powershell -File art-pipeline\lora\train.ps1

$env:PYTHONIOENCODING = "utf-8"
$lora = "C:\Users\nars7\Projects\aniML-vision\art-pipeline\lora"
& "$lora\venv\Scripts\python.exe" "$lora\sd-scripts\sdxl_train_network.py" `
  --pretrained_model_name_or_path "$lora\base\sd_xl_base_1.0.safetensors" `
  --dataset_config "$lora\dataset.toml" `
  --output_dir "$lora\output" `
  --output_name "animl_style_v6" `
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
  --max_token_length 225 `
  --max_train_steps 1500 `
  --save_every_n_steps 250 `
  --sample_every_n_steps 250 `
  --sample_prompts "$lora\sample_prompts.txt" `
  --sample_sampler euler_a `
  --sdpa `
  --mixed_precision bf16 `
  --save_precision bf16 `
  --gradient_checkpointing `
  --cache_latents `
  --cache_latents_to_disk `
  --seed 42
