"""
Filter images that are too small (low resolution)
Catches tiny crops that the blur filter misses
"""
import cv2
import os
import shutil
from tqdm import tqdm

DATA_DIR = 'training_data_cropped'
QUARANTINE_DIR = 'quarantine_tiny'
MIN_WIDTH = 100  # Minimum width in pixels
MIN_HEIGHT = 100  # Minimum height in pixels

def filter_tiny_images(min_width=MIN_WIDTH, min_height=MIN_HEIGHT, dry_run=True):
    print("="*70)
    print("TINY IMAGE FILTER")
    print("="*70)
    print(f"Minimum size: {min_width}x{min_height} pixels")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("="*70)
    
    tiny_images = []
    
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
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        tiny_images.append({'path': img_path, 'width': 0, 'height': 0, 'species': species, 'split': split})
                        continue
                    
                    h, w = img.shape[:2]
                    
                    if w < min_width or h < min_height:
                        tiny_images.append({'path': img_path, 'width': w, 'height': h, 'species': species, 'split': split})
                        
                except Exception as e:
                    print(f"Error: {e}")
    
    # Report
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Tiny images found: {len(tiny_images)}")
    
    if tiny_images:
        tiny_images.sort(key=lambda x: x['width'] * x['height'])
        
        print("\nSMALLEST 20:")
        for img in tiny_images[:20]:
            print(f"  {img['species']}/{os.path.basename(img['path'])}: {img['width']}x{img['height']}")
        
        from collections import defaultdict
        species_counts = defaultdict(int)
        for img in tiny_images:
            species_counts[img['species']] += 1
        
        print("\nMost affected species:")
        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {species}: {count}")
        
        if not dry_run:
            confirm = input(f"\nQuarantine {len(tiny_images)} tiny images? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                os.makedirs(QUARANTINE_DIR, exist_ok=True)
                
                for img in tiny_images:
                    src = img['path']
                    dest_dir = os.path.join(QUARANTINE_DIR, img['species'], img['split'])
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    shutil.move(src, os.path.join(dest_dir, os.path.basename(src)))
                    
                    json_src = os.path.splitext(src)[0] + '.json'
                    if os.path.exists(json_src):
                        shutil.move(json_src, os.path.join(dest_dir, os.path.basename(json_src)))
                
                print(f"\n✓ Quarantined {len(tiny_images)} images")
            else:
                print("Cancelled")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-size', type=int, default=100, help='Min width/height (default: 100)')
    parser.add_argument('--live', action='store_true', help='Actually move files')
    args = parser.parse_args()
    
    filter_tiny_images(args.min_size, args.min_size, dry_run=not args.live)
