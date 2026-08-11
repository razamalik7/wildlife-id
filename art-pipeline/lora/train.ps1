# AnimlVision style LoRA - SDXL, RTX 5080 (16GB)
# Recipe per renovation plan: rank 32 / alpha 16, LR 1e-4, ~1500 steps,
# content-only captions, UNet only. Checkpoints + samples every 250 steps.
#
# Run AFTER the GPU is free:  powershell -File art-pipeline\lora\train.ps1

$lora = "C:\Users\nars7\Projects\aniML-vision\art-pipeline\lora"
& "$lora\venv\Scripts\python.exe" "$lora\sd-scripts\sdxl_train_network.py" `
  --pretrained_model_name_or_path "$lora\base\sd_xl_base_1.0.safetensors" `
  --dataset_config "$lora\dataset.toml" `
  --output_dir "$lora\output" `
  --output_name "animl_style_v3" `
  --save_model_as safetensors `
  --network_module networks.lora `
  --network_dim 32 `
  --network_alpha 16 `
  --network_train_unet_only `
  --learning_rate 1e-4 `
  --lr_scheduler cosine `
  --lr_warmup_steps 100 `
  --optimizer_type AdamW8bit `
  --max_train_steps 2500 `
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
