"""
Optimized dataset balancing script
- Uses sqrt-sampling instead of max-matching (less duplication)
- Supports both training_data and training_data_cropped
- Shows before/after stats
- Handles metadata JSON properly
- Progress indicator
"""
import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# CONFIG
DATA_DIR = 'training_data_cropped'  # Updated for cropped dataset
BALANCE_MODE = 'sqrt'  # 'sqrt' (recommended), 'max', or 'median'

def get_class_counts(train_dir):
    """Get image counts per class (excluding clones)"""
    counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            # Count only original images (not clones)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                     and '_clone_' not in f]
            counts[class_name] = len(images)
    return counts

def remove_old_clones(train_dir):
    """Remove previously created clones before rebalancing"""
    print("Removing old clones...")
    removed = 0
    for class_name in tqdm(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for f in os.listdir(class_path):
            if '_clone_' in f:
                os.remove(os.path.join(class_path, f))
                removed += 1
    
    print(f"  Removed {removed} old clones")
    return removed

def balance_classes():
    train_dir = os.path.join(DATA_DIR, 'train')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found")
        return
    
    print("="*70)
    print("DATASET BALANCING")
    print("="*70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Balance mode: {BALANCE_MODE}")
    print("="*70)
    
    # Remove old clones first
    remove_old_clones(train_dir)
    
    # Get class counts
    counts = get_class_counts(train_dir)
    
    # Calculate target count based on mode
    count_values = list(counts.values())
    
    if BALANCE_MODE == 'max':
        target = max(count_values)
    elif BALANCE_MODE == 'median':
        target = int(np.median(count_values))
    elif BALANCE_MODE == 'sqrt':
        # Sqrt-smoothed: target = max * sqrt(min/max)
        # This reduces extreme duplication for small classes
        min_count = min(count_values)
        max_count = max(count_values)
        target = int(max_count * np.sqrt(min_count / max_count))
    else:
        target = max(count_values)
    
    print(f"\nBefore balancing:")
    print(f"  Classes: {len(counts)}")
    print(f"  Min images: {min(count_values)} ({min(counts, key=counts.get)})")
    print(f"  Max images: {max(count_values)} ({max(counts, key=counts.get)})")
    print(f"  Total images: {sum(count_values)}")
    print(f"\n🎯 Target count per class: {target}")
    
    # Calculate how many need to be added
    total_to_add = sum(max(0, target - c) for c in count_values)
    print(f"   Total clones to create: {total_to_add}")
    
    confirm = input("\nProceed with balancing? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled")
        return
    
    # Balance each class
    clones_created = 0
    
    for class_name in tqdm(counts.keys(), desc="Balancing"):
        class_path = os.path.join(train_dir, class_name)
        current = counts[class_name]
        
        if current >= target:
            continue
        
        # Get original images (not clones)
        original_images = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                          and '_clone_' not in f]
        
        needed = target - current
        
        for i in range(needed):
            # Pick random original to clone
            src_image = random.choice(original_images)
            src_path = os.path.join(class_path, src_image)
            
            # Create clone name
            name, ext = os.path.splitext(src_image)
            dst_name = f"{name}_clone_{i}{ext}"
            dst_path = os.path.join(class_path, dst_name)
            
            shutil.copy(src_path, dst_path)
            clones_created += 1
            
            # Clone metadata JSON if exists
            src_json = os.path.splitext(src_path)[0] + ".json"
            dst_json = os.path.splitext(dst_path)[0] + ".json"
            
            if os.path.exists(src_json):
                shutil.copy(src_json, dst_json)
    
    # Final stats
    new_counts = get_class_counts(train_dir)
    # Count including clones
    total_with_clones = sum(
        len([f for f in os.listdir(os.path.join(train_dir, c)) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for c in new_counts.keys()
    )
    
    print("\n" + "="*70)
    print("BALANCING COMPLETE")
    print("="*70)
    print(f"Clones created: {clones_created}")
    print(f"Total images now: {total_with_clones}")
    print(f"Target per class: {target}")
    print("✅ Dataset balanced!")

if __name__ == "__main__":
    balance_classes()