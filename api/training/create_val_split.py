"""
Split training images to create validation set for incomplete species.
Takes 70 images from each species' train folder and moves them to val folder.
"""
import os
import shutil
from pathlib import Path

DATA_DIR = 'training_data_v2'
VAL_TARGET = 70

# Species missing validation data
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
print("CREATING VALIDATION SETS FOR INCOMPLETE SPECIES")
print("="*70)

for species in INCOMPLETE_SPECIES:
    species_train = train_dir / species
    species_val = val_dir / species
    
    # Get all images (including augmented ones)
    train_images = sorted(list(species_train.glob('*.jpg')))
    
    if len(train_images) < VAL_TARGET:
        print(f"⚠️ {species}: Only {len(train_images)} images, cannot create val set")
        continue
    
    # Create val directory if needed
    species_val.mkdir(parents=True, exist_ok=True)
    
    # Move last 70 images to validation (these are likely augmented ones)
    to_move = train_images[-VAL_TARGET:]
    
    print(f"\n{species}:")
    print(f"  Moving {len(to_move)} images from train to val...")
    
    for img_path in to_move:
        # Move image
        dest_img = species_val / img_path.name
        shutil.move(str(img_path), str(dest_img))
        
        # Move corresponding JSON if exists
        json_path = img_path.with_suffix('.json')
        if json_path.exists():
            dest_json = species_val / json_path.name
            shutil.move(str(json_path), str(dest_json))
    
    # Verify
    remaining_train = len(list(species_train.glob('*.jpg')))
    new_val = len(list(species_val.glob('*.jpg')))
    
    print(f"  ✓ Train: {remaining_train}, Val: {new_val}")

print("\n" + "="*70)
print("✅ VALIDATION SPLIT COMPLETE")
print("="*70)
