import os
import time
from pathlib import Path

# Dynamic path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
TARGET_PER_SPECIES = 670

def monitor():
    print(f"Monitoring {DATA_DIR}...")
    while True:
        if not os.path.exists(DATA_DIR):
            print("Waiting for directory...")
            time.sleep(2)
            continue
            
        species_dirs = [d for d in os.listdir(os.path.join(DATA_DIR, 'train')) if os.path.isdir(os.path.join(DATA_DIR, 'train', d))]
        
        print("\n" + "="*50)
        print(f"DATASET STATUS ({len(species_dirs)} species found)")
        print("="*50)
        
        total_imgs = 0
        for species in species_dirs:
            train_dir = os.path.join(DATA_DIR, 'train', species)
            val_dir = os.path.join(DATA_DIR, 'val', species)
            
            n_train = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
            n_val = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')]) if os.path.exists(val_dir) else 0
            
            total = n_train + n_val
            pct = (total / TARGET_PER_SPECIES) * 100
            
            bar_len = 20
            filled = int(bar_len * (total / TARGET_PER_SPECIES))
            bar = "█" * filled + "-" * (bar_len - filled)
            
            print(f"{species[:20]:<20} | {bar} | {total}/{TARGET_PER_SPECIES} ({pct:.1f}%)")
            total_imgs += total
            
        print(f"\nTOTAL IMAGES: {total_imgs}")
        time.sleep(5)

if __name__ == "__main__":
    monitor()
