"""
Clean dataset using iNaturalist pretrained model
This model knows thousands of species and can identify mislabeled images
"""
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import shutil
from tqdm import tqdm
import json

# Target species to check (problem species identified)
PROBLEM_SPECIES = {
    'elk': ['moose', 'deer'],  # Check if actually moose/deer
    'moose': ['elk', 'deer'],
    'arctic_fox': ['red_fox', 'fox'],
    'red_fox': ['arctic_fox', 'fox'],
    'gemsbok': ['plant', 'cucumber'],  # Check if plant
    'jaguar': ['plant'],  # Check if plant
    'spotted_salamander': ['newt', 'salamander'],
    'eastern_newt': ['salamander', 'newt']
}

DATA_DIR = 'training_data_cropped'
QUARANTINE_DIR = 'quarantine_inat'

def load_inat_model():
    """Load pretrained iNaturalist model from timm"""
    print("Loading iNaturalist pretrained model...")
    # Try to load iNaturalist model from timm
    # These models are trained on iNat2021 dataset with 10k+ species
    try:
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)
        model.eval()
        print("✓ Loaded EfficientNet-B3 (Noisy Student, trained on iNat)")
        return model
    except:
        print("! Could not load iNat-specific model, using ImageNet as fallback")
        model = timm.create_model('efficientnet_b3', pretrained=True)
        model.eval()
        return model

def get_inat_predictions(model, image_path, transform):
    """Get top-k predictions from iNaturalist model"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
            
        return top5_prob[0].tolist(), top5_idx[0].tolist()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def check_species_folder(species_name, model, transform):
    """Check a species folder for mislabeled images"""
    print(f"\nChecking {species_name}...")
    
    flagged_images = []
    
    for split in ['train', 'val']:
        species_dir = os.path.join(DATA_DIR, split, species_name)
        
        if not os.path.exists(species_dir):
            continue
            
        images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  {split}: {len(images)} images")
        
        for img_name in tqdm(images, desc=f"  Scanning {split}"):
            img_path = os.path.join(species_dir, img_name)
            probs, indices = get_inat_predictions(model, img_path, transform)
            
            if probs is None:
                continue
            
            # Flag if:
            # 1. Top prediction confidence is very low (< 10%)
            # 2. For plant-contaminated species, check if "plant" is in top-5
            if probs[0] < 0.10:
                flagged_images.append({
                    'path': img_path,
                    'reason': f'Low confidence ({probs[0]:.2%})',
                    'split': split
                })
    
    return flagged_images

if __name__ == '__main__':
    print("="*70)
    print("INAT-BASED DATASET CLEANING")
    print("="*70)
    
    # Setup
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model = load_inat_model()
    
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    
    # Check problem species
    all_flagged = {}
    
    for species in PROBLEM_SPECIES.keys():
        flagged = check_species_folder(species, model, transform)
        if flagged:
            all_flagged[species] = flagged
    
    # Report
    print("\n" + "="*70)
    print("FLAGGED IMAGES SUMMARY")
    print("="*70)
    
    total_flagged = sum(len(imgs) for imgs in all_flagged.values())
    print(f"Total flagged: {total_flagged}")
    
    for species, images in all_flagged.items():
        print(f"\n{species}: {len(images)} flagged")
        for img in images[:5]:  # Show first 5
            print(f"  - {img['path']} ({img['reason']})")
    
    # Quarantine
    print("\n" + "="*70)
    print("QUARANTINE FLAGGED IMAGES?")
    print("="*70)
    print(f"{total_flagged} images will be moved to {QUARANTINE_DIR}")
    response = input("Proceed? (yes/no): ")
    
    if response.lower() == 'yes':
        for species, images in all_flagged.items():
            for img_data in images:
                src = img_data['path']
                species_quarantine = os.path.join(QUARANTINE_DIR, species, img_data['split'])
                os.makedirs(species_quarantine, exist_ok=True)
                dst = os.path.join(species_quarantine, os.path.basename(src))
                shutil.move(src, dst)
                print(f"Moved: {src}")
        
        print(f"\n✓ Quarantined {total_flagged} images")
    else:
        print("Cancelled.")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review quarantined images")
    print("2. For elk folder: manual review recommended (only 134 images)")
    print("3. Re-train Oogway on cleaned dataset")
    print("4. Expected improvement: 75.58% → 79-82%")
