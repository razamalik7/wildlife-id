"""
FIX VALIDATION SPLIT - Proper approach for rare species
Instead of MOVING images from train to val (which reduces training data),
we COPY originals from train and AUGMENT them for validation.

This preserves all training data while creating a separate validation set.
"""
import os
import json
import random
from pathlib import Path
from PIL import Image, ImageEnhance

DATA_DIR = 'training_data_v2'
VAL_TARGET = 70

# Apply same augmentation as balance_dataset_v2.py
def apply_augmentation(image):
    img = image.copy()
    
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    brightness_factor = random.uniform(0.85, 1.15)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    
    contrast_factor = random.uniform(0.85, 1.15)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    color_factor = random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color_factor)
    
    return img

# Species that need validation data
INCOMPLETE_SPECIES = [
    'black-footed_ferret',
    'eastern_hellbender', 
    'veiled_chameleon',
    'walrus',
    'wolverine'
]

train_dir = Path(DATA_DIR) / 'train'
val_dir = Path(DATA_DIR) / 'val'

print("="*70)
print("FIXING VALIDATION SPLIT - COPY & AUGMENT APPROACH")
print("="*70)
print("Strategy: Copy originals from train, augment for val (preserves training data)")
print("="*70)

for species in INCOMPLETE_SPECIES:
    species_train = train_dir / species
    species_val = val_dir / species
    
    print(f"\n{species}:")
    
    # Step 1: Move back any existing val images to train
    existing_val = list(species_val.glob('*.jpg'))
    if existing_val:
        print(f"  Moving {len(existing_val)} images back from val to train...")
        for img_path in existing_val:
            dest = species_train / img_path.name
            img_path.rename(dest)
            
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                dest_json = species_train / json_path.name
                json_path.rename(dest_json)
    
    # Step 2: Get ORIGINAL images (not augmented) from train
    train_images = [f for f in species_train.glob('*.jpg') 
                    if '_aug_' not in f.name and '_clone_' not in f.name]
    
    print(f"  Found {len(train_images)} original images in train")
    
    if len(train_images) < VAL_TARGET:
        print(f"  ⚠️ Not enough originals - will create {VAL_TARGET} val images from {len(train_images)} sources")
    
    # Step 3: Create validation set by COPYING and AUGMENTING train originals
    species_val.mkdir(parents=True, exist_ok=True)
    
    print(f"  Creating {VAL_TARGET} augmented validation images...")
    
    for i in range(VAL_TARGET):
        # Pick random original
        src_img_path = random.choice(train_images)
        
        try:
            # Load and augment
            img = Image.open(src_img_path).convert('RGB')
            aug_img = apply_augmentation(img)
            
            # Save to val with new name
            val_img_name = f"val_aug_{i:03d}.jpg"
            val_img_path = species_val / val_img_name
            aug_img.save(val_img_path, quality=95)
            
            # Copy and modify metadata
            src_json_path = src_img_path.with_suffix('.json')
            if src_json_path.exists():
                with open(src_json_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['augmented'] = True
                metadata['source_image'] = src_img_path.name
                metadata['augmentation_type'] = 'validation'
                metadata['split'] = 'val'
                
                val_json_path = val_img_path.with_suffix('.json')
                with open(val_json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        except Exception as e:
            print(f"    Error creating val image {i}: {e}")
    
    # Verify
    final_train = len(list(species_train.glob('*.jpg')))
    final_val = len(list(species_val.glob('*.jpg')))
    print(f"  ✓ Final: Train={final_train}, Val={final_val}")

print("\n" + "="*70)
print("✅ VALIDATION FIX COMPLETE")
print("="*70)
