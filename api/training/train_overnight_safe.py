"""
Safe Overnight Training - Both Models
======================================
Run both B3 and ConvNeXt with very low LR for overnight training.
"""

import subprocess
import sys
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def main():
    log("🌙 SAFE OVERNIGHT TRAINING")
    log("=" * 60)
    log("Strategy: Very low LR (1e-5), no hard negatives, gentle mixup")
    log("=" * 60)
    
    # Model 1: B3
    log("\n📦 Model 1/2: EfficientNet-B3")
    result1 = subprocess.run([
        sys.executable, "finetune_safe.py",
        "--model", "oogway_b3_best.pth",
        "--type", "b3",
        "--roots", "training_data_v2", "training_data_refined",
        "--epochs", "10",
        "--lr", "1e-5"
    ])
    
    if result1.returncode == 0:
        log("✅ B3 Complete!")
    else:
        log(f"❌ B3 Failed (exit {result1.returncode})")
    
    log("=" * 60)
    
    # Model 2: ConvNeXt
    log("\n📦 Model 2/2: ConvNeXt-Tiny")
    result2 = subprocess.run([
        sys.executable, "finetune_safe.py",
        "--model", "oogway_convnext_final.pth",
        "--type", "convnext",
        "--roots", "training_data_v2", "training_data_refined",
        "--epochs", "10",
        "--lr", "1e-5"
    ])
    
    if result2.returncode == 0:
        log("✅ ConvNeXt Complete!")
    else:
        log(f"❌ ConvNeXt Failed (exit {result2.returncode})")
    
    log("=" * 60)
    log("🏆 OVERNIGHT TRAINING COMPLETE")
    log("Check for:")
    log("  - oogway_b3_best_safe_finetuned.pth")
    log("  - oogway_convnext_final_safe_finetuned.pth")

if __name__ == "__main__":
    main()
