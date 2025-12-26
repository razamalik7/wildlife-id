"""
Queue Oogway Training - Waits for scraping to complete, then starts training
Run this and go to sleep! Training will start automatically when data is ready.
"""
import os
import time
import subprocess
import sys
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'training_data_v2', 'train')
VAL_DIR = os.path.join(SCRIPT_DIR, 'training_data_v2', 'val')
TARGET_SPECIES = 100
TARGET_IMAGES_IN_TRAIN = 600  # ~90% of 670 goes to train
CHECK_INTERVAL_SECONDS = 60  # Check every minute

def count_species_status():
    """Count how many species are complete (600+ images in train folder)"""
    if not os.path.exists(TRAIN_DIR):
        return 0, 0, 0
    
    species_dirs = [d for d in os.listdir(TRAIN_DIR) 
                    if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    complete = 0
    total_images = 0
    
    for species in species_dirs:
        train_path = os.path.join(TRAIN_DIR, species)
        img_count = len([f for f in os.listdir(train_path) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))])
        total_images += img_count
        if img_count >= TARGET_IMAGES_IN_TRAIN:
            complete += 1
    
    return complete, len(species_dirs), total_images

def main():
    print("=" * 60)
    print("🐢 OOGWAY TRAINING QUEUE")
    print("=" * 60)
    print(f"Waiting for {TARGET_SPECIES} species with {TARGET_IMAGES_IN_TRAIN}+ images in train folder")
    print(f"Checking every {CHECK_INTERVAL_SECONDS} seconds...")
    print("=" * 60)
    print()
    
    while True:
        complete, total_dirs, total_images = count_species_status()
        now = datetime.now().strftime("%H:%M:%S")
        
        progress = complete / TARGET_SPECIES * 100
        print(f"[{now}] {complete}/{TARGET_SPECIES} species complete ({progress:.1f}%) | {total_images:,} total images")
        
        if complete >= TARGET_SPECIES:
            print()
            print("=" * 60)
            print("✅ ALL SPECIES COMPLETE! Starting Oogway training...")
            print("=" * 60)
            print()
            break
        
        time.sleep(CHECK_INTERVAL_SECONDS)
    
    # Start training
    train_script = os.path.join(SCRIPT_DIR, 'train_oogway.py')
    print(f"Executing: python {train_script}")
    print()
    
    # Run training (replaces current process)
    os.execv(sys.executable, [sys.executable, train_script])

if __name__ == "__main__":
    main()
