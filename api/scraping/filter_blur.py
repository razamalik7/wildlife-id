"""
Blur detection filter for cropped dataset
Uses Laplacian variance to detect blurry images
"""
import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np

DATA_DIR = 'training_data_cropped'
QUARANTINE_DIR = 'quarantine_blur'
BLUR_THRESHOLD = 100  # Images with variance below this are considered blurry

def calculate_blur_score(image_path):
    """
    Calculate blur score using Laplacian variance
    Lower score = more blurry
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def filter_blurry_images(threshold=BLUR_THRESHOLD, dry_run=True):
    """
    Scan dataset and remove blurry images
    """
    print("="*70)
    print("BLUR DETECTION FILTER")
    print("="*70)
    print(f"Threshold: {threshold} (lower = blurrier)")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE - will quarantine'}")
    print("="*70)
    
    blurry_images = []
    all_scores = []
    
    for split in ['train', 'val']:
        split_dir = os.path.join(DATA_DIR, split)
        
        if not os.path.exists(split_dir):
            continue
        
        species_list = sorted(os.listdir(split_dir))
        
        for species in tqdm(species_list, desc=f"Scanning {split}"):
            species_dir = os.path.join(split_dir, species)
            
            if not os.path.isdir(species_dir):
                continue
            
            images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_name in images:
                img_path = os.path.join(species_dir, img_name)
                score = calculate_blur_score(img_path)
                all_scores.append(score)
                
                if score < threshold:
                    blurry_images.append({
                        'path': img_path,
                        'score': score,
                        'species': species,
                        'split': split
                    })
    
    # Report
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    if all_scores:
        print(f"Total images scanned: {len(all_scores)}")
        print(f"Score range: {min(all_scores):.1f} - {max(all_scores):.1f}")
        print(f"Mean score: {np.mean(all_scores):.1f}")
        print(f"Median score: {np.median(all_scores):.1f}")
    
    print(f"\nBlurry images found: {len(blurry_images)}")
    
    if blurry_images:
        # Show worst 20
        blurry_images.sort(key=lambda x: x['score'])
        
        print("\nWORST 20 (most blurry):")
        for img in blurry_images[:20]:
            print(f"  {img['species']}/{os.path.basename(img['path'])}: {img['score']:.1f}")
        
        # Species breakdown
        from collections import defaultdict
        species_counts = defaultdict(int)
        for img in blurry_images:
            species_counts[img['species']] += 1
        
        print("\nMost affected species:")
        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {species}: {count} blurry")
        
        if not dry_run:
            print("\n" + "="*70)
            confirm = input(f"Quarantine {len(blurry_images)} blurry images? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                os.makedirs(QUARANTINE_DIR, exist_ok=True)
                
                for img in blurry_images:
                    src = img['path']
                    dest_dir = os.path.join(QUARANTINE_DIR, img['species'], img['split'])
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    shutil.move(src, os.path.join(dest_dir, os.path.basename(src)))
                    
                    # Also move the JSON metadata if exists
                    json_src = os.path.splitext(src)[0] + '.json'
                    if os.path.exists(json_src):
                        shutil.move(json_src, os.path.join(dest_dir, os.path.basename(json_src)))
                
                print(f"\n✓ Quarantined {len(blurry_images)} images to {QUARANTINE_DIR}")
            else:
                print("Cancelled")
    else:
        print("\nNo blurry images found!")
    
    return blurry_images

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=BLUR_THRESHOLD, 
                       help='Blur threshold (default: 100)')
    parser.add_argument('--live', action='store_true',
                       help='Actually move files (default: dry run)')
    args = parser.parse_args()
    
    filter_blurry_images(threshold=args.threshold, dry_run=not args.live)
