"""
Enhanced dataset balancing with proper augmentation
- Applies transformations (rotation, flip, brightness, contrast) instead of simple cloning
- Works with training_data_v2
- Balances both train and val directories
- Preserves and copies metadata JSON files
- Handles rare species with limited data
"""
import os
import json
import shutil
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageEnhance

# CONFIG
DATA_DIR = 'training_data_v2'
BALANCE_MODE = 'sqrt'  # 'sqrt' (recommended), 'max', or 'median'
TRAIN_TARGET = 600
VAL_TARGET = 70
USE_AUGMENTATION = True  # If False, falls back to simple cloning

def apply_augmentation(image):
    """
    Apply random augmentation to an image.
    Returns augmented PIL Image.
    """
    img = image.copy()
    
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # Random brightness (0.85 to 1.15)
    brightness_factor = random.uniform(0.85, 1.15)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    
    # Random contrast (0.85 to 1.15)
    contrast_factor = random.uniform(0.85, 1.15)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Random color saturation (0.9 to 1.1)
    color_factor = random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color_factor)
    
    return img

def get_class_counts(split_dir):
    """Get image counts per class (excluding augmented/cloned images)"""
    counts = {}
    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if os.path.isdir(class_path):
            # Count only original images (not clones or augmented)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                     and '_clone_' not in f 
                     and '_aug_' not in f]
            counts[class_name] = len(images)
    return counts

def remove_old_augmentations(split_dir):
    """Remove previously created augmented/cloned images"""
    print(f"Removing old augmentations from {os.path.basename(split_dir)}...")
    removed = 0
    for class_name in tqdm(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for f in os.listdir(class_path):
            if '_clone_' in f or '_aug_' in f:
                # Remove both image and JSON
                file_path = os.path.join(class_path, f)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed += 1
                    
                    # Remove corresponding JSON if exists
                    json_path = os.path.splitext(file_path)[0] + '.json'
                    if os.path.exists(json_path):
                        os.remove(json_path)
    
    print(f"  Removed {removed} old augmented/cloned images")
    return removed

def balance_split(split_dir, target_count, split_name):
    """Balance a single split (train or val) directory"""
    
    print(f"\n{'='*70}")
    print(f"BALANCING {split_name.upper()}")
    print(f"{'='*70}")
    
    # Remove old augmentations first
    remove_old_augmentations(split_dir)
    
    # Get class counts
    counts = get_class_counts(split_dir)
    
    count_values = list(counts.values())
    
    print(f"\nBefore balancing:")
    print(f"  Classes: {len(counts)}")
    print(f"  Min images: {min(count_values)} ({min(counts, key=counts.get)})")
    print(f"  Max images: {max(count_values)} ({max(counts, key=counts.get)})")
    print(f"  Total images: {sum(count_values)}")
    print(f"  Target: {target_count}")
    
    # Calculate how many need to be added
    classes_needing_balance = {k: v for k, v in counts.items() if v < target_count}
    total_to_add = sum(target_count - c for c in classes_needing_balance.values())
    
    print(f"\n  Classes needing augmentation: {len(classes_needing_balance)}")
    print(f"  Total images to create: {total_to_add}")
    
    if total_to_add == 0:
        print(f"  ✅ {split_name} already balanced!")
        return
    
    # Balance each class
    augmented_created = 0
    
    for class_name in tqdm(counts.keys(), desc=f"Balancing {split_name}"):
        class_path = os.path.join(split_dir, class_name)
        current = counts[class_name]
        
        if current >= target_count:
            continue
        
        # Get original images (not augmented or cloned)
        original_images = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                          and '_clone_' not in f
                          and '_aug_' not in f]
        
        if not original_images:
            print(f"  ⚠️ No original images found for {class_name}, skipping")
            continue
        
        needed = target_count - current
        
        for i in range(needed):
            # Pick random original to augment
            src_image = random.choice(original_images)
            src_path = os.path.join(class_path, src_image)
            
            # Create augmented image
            try:
                img = Image.open(src_path).convert('RGB')
                
                if USE_AUGMENTATION:
                    aug_img = apply_augmentation(img)
                    prefix = 'aug'
                else:
                    aug_img = img
                    prefix = 'clone'
                
                # Create destination filename
                name, ext = os.path.splitext(src_image)
                dst_name = f"{name}_{prefix}_{i:03d}{ext}"
                dst_path = os.path.join(class_path, dst_name)
                
                # Save augmented image
                aug_img.save(dst_path, quality=95)
                augmented_created += 1
                
                # Copy and update metadata JSON
                src_json = os.path.splitext(src_path)[0] + ".json"
                dst_json = os.path.splitext(dst_path)[0] + ".json"
                
                if os.path.exists(src_json):
                    with open(src_json, 'r') as f:
                        metadata = json.load(f)
                    
                    # Mark as augmented in metadata
                    metadata['augmented'] = True
                    metadata['source_image'] = src_image
                    metadata['augmentation_type'] = 'transformation' if USE_AUGMENTATION else 'clone'
                    
                    with open(dst_json, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
            except Exception as e:
                print(f"  ⚠️ Error augmenting {src_image}: {e}")
                continue
    
    # Final stats
    final_counts = {}
    for class_name in counts.keys():
        class_path = os.path.join(split_dir, class_name)
        total_images = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        final_counts[class_name] = total_images
    
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} BALANCING COMPLETE")
    print(f"{'='*70}")
    print(f"Images created: {augmented_created}")
    print(f"Total images now: {sum(final_counts.values())}")
    print(f"Classes at target ({target_count}): {sum(1 for c in final_counts.values() if c >= target_count)}/{len(final_counts)}")

def main():
    print("="*70)
    print("ENHANCED DATASET BALANCING WITH AUGMENTATION")
    print("="*70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Augmentation: {'ENABLED' if USE_AUGMENTATION else 'DISABLED (simple cloning)'}")
    print(f"Targets: {TRAIN_TARGET} train, {VAL_TARGET} val")
    print("="*70)
    
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found")
        return
    
    if not os.path.exists(val_dir):
        print(f"Error: {val_dir} not found")
        return
    
    # Balance both train and val
    balance_split(train_dir, TRAIN_TARGET, 'train')
    balance_split(val_dir, VAL_TARGET, 'val')
    
    print("\n" + "="*70)
    print("✅ DATASET BALANCING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
