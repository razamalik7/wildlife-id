"""
Overnight Training Script - Trains Both Models Sequentially
============================================================
Run this to train both B3 and ConvNeXt overnight.

Usage:
  python train_overnight.py
"""

import subprocess
import sys
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def main():
    log("🌙 OVERNIGHT TRAINING STARTED")
    log("="*60)
    
    # Model 1: EfficientNet-B3
    log("📦 Training Model 1/2: EfficientNet-B3")
    result1 = subprocess.run([
        sys.executable, "targeted_finetune_v2.py",
        "--model", "oogway_b3_best.pth",
        "--type", "b3",
        "--roots", "training_data_v2", "training_data_refined",
        "--epochs", "15"
    ], cwd=".")
    
    if result1.returncode == 0:
        log("✅ B3 Training Complete!")
    else:
        log(f"❌ B3 Training Failed (exit code {result1.returncode})")
    
    log("="*60)
    
    # Model 2: ConvNeXt
    log("📦 Training Model 2/2: ConvNeXt-Tiny")
    result2 = subprocess.run([
        sys.executable, "targeted_finetune_v2.py",
        "--model", "oogway_convnext_final.pth",
        "--type", "convnext",
        "--roots", "training_data_v2", "training_data_refined",
        "--epochs", "15"
    ], cwd=".")
    
    if result2.returncode == 0:
        log("✅ ConvNeXt Training Complete!")
    else:
        log(f"❌ ConvNeXt Training Failed (exit code {result2.returncode})")
    
    log("="*60)
    log("🏆 OVERNIGHT TRAINING FINISHED")
    log("Models saved as:")
    log("  - oogway_b3_best_finetuned.pth")
    log("  - oogway_convnext_final_finetuned.pth")

if __name__ == "__main__":
    main()
