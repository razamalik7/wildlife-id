# Generate the checkpoint x weight comparison grid under real conditions
# (framing-contract prompts + negatives). Run after training completes.
#   powershell -File eval_gen.ps1
$lora = "C:\Users\nars7\Projects\aniML-vision\art-pipeline\lora"
$ckpts = @(
    @{file = "animl_style_v1-step00000500.safetensors"; tag = "s0500"},
    @{file = "animl_style_v1-step00001000.safetensors"; tag = "s1000"},
    @{file = "animl_style_v1.safetensors";               tag = "s1500"}
)
foreach ($c in $ckpts) {
    foreach ($w in @(0.8, 1.0)) {
        & "$lora\venv\Scripts\python.exe" "$lora\sd-scripts\sdxl_gen_img.py" `
            --ckpt "$lora\base\sd_xl_base_1.0.safetensors" `
            --outdir "$lora\output\eval\$($c.tag)_w$w" `
            --network_module networks.lora `
            --network_weights "$lora\output\$($c.file)" `
            --network_mul $w `
            --from_file "$lora\eval_prompts.txt" `
            --W 1024 --H 1024 --steps 28 --sampler euler_a `
            --seed 42 --bf16 --sdpa --batch_size 1 --images_per_prompt 1
    }
}
