# Photo-reference generation test: LoRA (style) + img2img photo (anatomy/composition)
# + framing-contract prompt. Sweeps denoise strength - the plan's "main dial".
$lora = "C:\Users\nars7\Projects\aniML-vision\art-pipeline\lora"
$species = @(
    @{slug = "wood_duck";      prompt = "a wood duck, full body wildlife illustration, single animal, painterly brushwork, plain white background, no scenery, no props, no cast shadow"},
    @{slug = "muskrat";        prompt = "a muskrat, full body wildlife illustration, single animal, painterly brushwork, long thin hairless tail, plain white background, no scenery, no props, no cast shadow"},
    @{slug = "american_robin"; prompt = "an american robin, full body wildlife illustration, single animal, painterly brushwork, plain white background, no scenery, no props, no cast shadow"}
)
foreach ($s in $species) {
    foreach ($str in @(0.6, 0.7, 0.8)) {
        & "$lora\venv\Scripts\python.exe" "$lora\sd-scripts\sdxl_gen_img.py" `
            --ckpt "$lora\base\sd_xl_base_1.0.safetensors" `
            --outdir "$lora\output\img2img_cutout\$($s.slug)_str$str" `
            --network_module networks.lora `
            --network_weights "$lora\output\animl_style_v1.safetensors" `
            --network_mul 1.0 `
            --prompt "$($s.prompt) --n photo, photorealistic, pattern, multiple animals, text, watermark, scenery, glow, vignette, shadow" `
            --image_path "$lora\eval_refs\$($s.slug)_cutout1024.png" `
            --strength $str `
            --W 1024 --H 1024 --steps 32 --sampler euler_a --scale 7 `
            --seed 42 --bf16 --sdpa --batch_size 1 --images_per_prompt 1
    }
}
