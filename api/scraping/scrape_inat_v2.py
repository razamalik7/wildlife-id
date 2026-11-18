"""
scrape_inat_v2.py - Improved iNaturalist Scraper with Quality Checks

Built-in quality filters:
1. Uses exact taxon_id from species_config.json (no name-based search)
2. Blur detection (Laplacian variance > 200)
3. Minimum resolution after crop (100x100)
4. YOLO animal detection (skip if no animal found)
5. Skip observation notes mentioning "dead" or "roadkill"
6. Quality grade = "research"
7. Target: 670 good images per species (600 train, 70 val)

Usage:
    python scrape_inat_v2.py                    # Scrape all species
    python scrape_inat_v2.py --species moose    # Scrape specific species
    python scrape_inat_v2.py --resume           # Resume from last progress
"""

import os
import json
import time
import random
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available - skipping crop validation")
    YOLO_AVAILABLE = False

# ============== CONFIG ==============
OUTPUT_DIR = 'training_data_v2'
TRAIN_COUNT = 600  # Images per species for training
VAL_COUNT = 70     # Images per species for validation
TOTAL_TARGET = TRAIN_COUNT + VAL_COUNT  # 670 per species

# Tiered quality - 85% high quality, 15% challenging
HIGH_QUALITY_RATIO = 0.85
CHALLENGING_RATIO = 0.15

# HIGH QUALITY thresholds (close-ups, sharp)
HQ_BLUR_THRESHOLD = 200    # Minimum Laplacian variance
HQ_MIN_SIZE = 100          # Minimum dimension after crop (pixels)

# CHALLENGING thresholds (distant silhouettes, harder images)
CH_BLUR_THRESHOLD = 80     # Lower blur threshold
CH_MIN_SIZE = 80           # Slightly smaller size (2.8x upscale to 224px)
CONF_THRESHOLD = 0.3       # YOLO confidence threshold
MARGIN_PCT = 0.15          # Crop margin percentage

# Animal classes in YOLO (COCO)
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

# iNaturalist API
INAT_API = 'https://api.inaturalist.org/v1'
PER_PAGE = 200  # Max per request
MAX_PAGES = 30  # Max pages to fetch (200 * 30 = 6000 candidates)

# Rate limiting
REQUEST_DELAY = 0.3  # Seconds between API requests
DOWNLOAD_DELAY = 0.1  # Seconds between image downloads

# ============== FUNCTIONS ==============

def load_species_config():
    """Load species configuration with taxon_ids"""
    with open('species_config.json', 'r') as f:
        config = json.load(f)
    
    species_map = {}
    for entry in config:
        name = entry['name']
        folder_name = name.lower().replace(' ', '_').replace('-', '-')
        taxon_id = entry.get('taxonomy', {}).get('taxon_id')
        
        if taxon_id:
            species_map[folder_name] = {
                'name': name,
                'taxon_id': taxon_id,
                'folder': folder_name
            }
        else:
            print(f"⚠️ Missing taxon_id for {name}")
    
    return species_map


def load_existing_ids(folder_name):
    """
    Scan all JSON files in train/val directories for this species 
    and return a set of already downloaded observation IDs.
    """
    seen_ids = set()
    dirs_to_check = [
        Path(OUTPUT_DIR) / 'train' / folder_name,
        Path(OUTPUT_DIR) / 'val' / folder_name
    ]
    
    print(f"  Scanning existing files for {folder_name}...")
    for d in dirs_to_check:
        if not d.exists(): continue
        # Use simple glob for reliability
        for f in d.glob('*.json'):
            try:
                # We need to read content to find obs_id if filename is just an index
                with open(f, 'r') as jf:
                    data = json.load(jf)
                    obs_id = data.get('observation_id')
                    if obs_id:
                        seen_ids.add(int(obs_id))
            except:
                pass
                
    print(f"  ✓ Found {len(seen_ids)} existing observations (Deduplication Active)")
    return seen_ids



# ============== PARALLEL PROCESSING HELPER ==============

def download_image_simple(url):
    """Simple download returning PIL Image or None"""
    try:
        url = url.replace('square', 'medium').replace('small', 'medium')
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except:
        return None

def process_batch_images(images, yolo_model):
    """
    Run YOLO on a batch of images and return crop boxes.
    Returns list of: None (skip), 'keep_full', or (x1,y1,x2,y2) (crop)
    """
    if not images:
        return []
        
    actions = []
    
    if YOLO_AVAILABLE and yolo_model:
        # Batch inference
        # Convert all to BGR numpy for YOLO
        imgs_cv = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
        
        # Run batch inference
        results = yolo_model(imgs_cv, verbose=False, conf=0.10, batch=len(images))
        
        wildlife_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        
        for i, res in enumerate(results):
            img = images[i]
            best_box = None
            max_conf = 0
            any_detection = len(res.boxes) > 0
            
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_id in wildlife_classes and conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0].tolist()
            
            if best_box:
                actions.append(('crop', best_box))
            elif not any_detection and min(img.size) < 400:
                actions.append(('skip', None)) # Likely evidence
            else:
                actions.append(('keep', None)) # Distant or misclassified
                
    else:
        # No YOLO, keep all valid images
        actions = [('keep', None)] * len(images)
        
    return actions

def calculate_blur_score(image):
    """Calculate blur score using Laplacian variance (higher = sharper)"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_dead_animal(observation):
    """Check if observation mentions dead animal"""
    dead_keywords = ['dead', 'roadkill', 'deceased', 'carcass', 'deceased', 'mortality']
    description = (observation.get('description') or '').lower()
    for field in observation.get('ofvs', []):
        if field.get('value', '').lower() in dead_keywords: return True
    for keyword in dead_keywords:
        if keyword in description: return True
    return False

def fetch_observations(taxon_id, page=1, per_page=200):
    """Fetch observations from iNaturalist API"""
    params = {
        'taxon_id': taxon_id,
        'quality_grade': 'research',
        'photos': 'true',
        'per_page': per_page,
        'page': page,
        'order': 'random'
    }
    try:
        r = requests.get(f'{INAT_API}/observations', params=params, timeout=30)
        if r.status_code == 200: return r.json().get('results', [])
        elif r.status_code == 429:
            print("Rate limited, waiting...")
            time.sleep(60)
            return fetch_observations(taxon_id, page, per_page)
        else:
            print(f"API error: {r.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching: {e}")
        return []

def get_quality_tier(image, blur, size):
    if blur >= HQ_BLUR_THRESHOLD and size >= HQ_MIN_SIZE:
        return 'high_quality'
    elif blur >= CH_BLUR_THRESHOLD and size >= CH_MIN_SIZE:
        return 'challenging'
    return None

def scrape_species(species_info, yolo_model, progress_file='scrape_progress.json'):
    """Scrape images for a single species using Batch Processing"""
    folder = species_info['folder']
    taxon_id = species_info['taxon_id']
    name = species_info['name']
    
    train_dir = Path(OUTPUT_DIR) / 'train' / folder
    val_dir = Path(OUTPUT_DIR) / 'val' / folder
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Init quotas
    hq_train_target = int(TRAIN_COUNT * HIGH_QUALITY_RATIO)
    ch_train_target = TRAIN_COUNT - hq_train_target
    hq_val_target = VAL_COUNT
    ch_val_target = 0
    
    # 1. Load Existing Counts & IDs (Deduplication)
    # ---------------------------------------------
    seen_ids = load_existing_ids(folder)
    
    # Recalculate current counts based on file tiers
    hq_train = 0; ch_train = 0
    hq_val = 0; ch_val = 0
    
    # Quick scan of directories for counts
    existing_train_files = list(train_dir.glob('*.json'))
    for f in existing_train_files:
        try:
            with open(f) as jf: 
                if json.load(jf).get('quality_tier') == 'high_quality': hq_train += 1
                else: ch_train += 1
        except: pass
            
    existing_val_files = list(val_dir.glob('*.json'))
    for f in existing_val_files:
        try:
            with open(f) as jf: 
                if json.load(jf).get('quality_tier') == 'high_quality': hq_val += 1
                else: ch_val += 1
        except: pass
    
    existing_train_total = hq_train + ch_train
    existing_val_total = hq_val + ch_val
    
    if existing_train_total >= TRAIN_COUNT and existing_val_total >= VAL_COUNT:
        print(f"  ✓ {name}: Already complete")
        return True

    print(f"  {name}: Need {TRAIN_COUNT - existing_train_total} train + {VAL_COUNT - existing_val_total} val")
    
    page = 1
    
    # Thread pool for downloads (Increased for speed, monitored for stability)
    MAX_WORKERS = 12 
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    pbar = tqdm(total=(TRAIN_COUNT + VAL_COUNT - existing_train_total - existing_val_total), desc=f"    {folder}", leave=False)
    
    # Batch size for processing (Memory Stability)
    BATCH_SIZE = 50
    
    while page <= MAX_PAGES:
        # Check quota
        if (hq_train + ch_train >= TRAIN_COUNT) and (hq_val + ch_val >= VAL_COUNT):
            break
            
        observations = fetch_observations(taxon_id, page)
        if not observations:
            break
            
        # Filter valid obs to process
        valid_obs = []
        urls = []
        
        for obs in observations:
            obs_id = obs.get('id')
            
            # DEDUPLICATION CHECK
            if obs_id in seen_ids: 
                continue # SKIP DUPLICATE
                
            if is_dead_animal(obs): continue
            
            photos = obs.get('photos', [])
            if not photos: continue
            url = photos[0].get('url')
            if not url: continue
            
            seen_ids.add(obs_id)
            valid_obs.append(obs)
            urls.append(url)
            
        if not valid_obs:
            page += 1
            continue
            
        # PROCESS IN BATCHES (Stability)
        # Instead of downloading all 200 at once, do chunks of BATCH_SIZE
        num_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for b in range(num_batches):
            start_idx = b * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            chunk_urls = urls[start_idx:end_idx]
            chunk_obs = valid_obs[start_idx:end_idx]
            
            # Parallel Download
            images = list(executor.map(download_image_simple, chunk_urls))
            
            # Filter failed downloads
            batch_obs = []
            batch_imgs = []
            for i, img in enumerate(images):
                if img and min(img.size) >= 100:
                    batch_obs.append(chunk_obs[i])
                    batch_imgs.append(img)
            
            if not batch_imgs:
                continue
                
            # Batch YOLO
            actions = process_batch_images(batch_imgs, yolo_model)
            
            # Process and Save
            for i, action in enumerate(actions):
                act_type, box = action
                if act_type == 'skip': continue
                    
                img = batch_imgs[i]
                obs = batch_obs[i]
                
                # Crop if needed
                if act_type == 'crop' and box:
                    x1, y1, x2, y2 = box
                    w, h = img.size
                    bw, bh = x2-x1, y2-y1
                    margin = MARGIN_PCT
                    nx1 = max(0, int(x1 - bw*margin))
                    ny1 = max(0, int(y1 - bh*margin))
                    nx2 = min(w, int(x2 + bw*margin))
                    ny2 = min(h, int(y2 + bh*margin))
                    final_img = img.crop((nx1, ny1, nx2, ny2))
                else:
                    final_img = img
                    
                # Check Quality
                if min(final_img.size) < CH_MIN_SIZE: continue
                    
                blur = calculate_blur_score(final_img)
                tier = get_quality_tier(final_img, blur, min(final_img.size))
                
                if not tier: continue
                    
                # Determine save location
                total_train = hq_train + ch_train
                total_val = hq_val + ch_val
                
                save_path = None
                
                if tier == 'high_quality':
                    if hq_train < hq_train_target and total_train < TRAIN_COUNT:
                        idx = total_train
                        save_path = train_dir / f"{idx:03d}"
                        hq_train += 1
                    elif hq_val < hq_val_target and total_val < VAL_COUNT:
                        idx = total_val
                        save_path = val_dir / f"{idx:03d}"
                        hq_val += 1
                elif tier == 'challenging':
                    if ch_train < ch_train_target and total_train < TRAIN_COUNT:
                        idx = total_train
                        save_path = train_dir / f"{idx:03d}"
                        ch_train += 1
                
                if save_path:
                    # Save
                    try:
                        final_img.save(f"{save_path}.jpg", quality=95)
                        metadata = {
                            'observation_id': obs['id'],
                            'taxon_id': taxon_id,
                            'species': name,
                            'quality_tier': tier,
                            'location': obs.get('location'),
                            'observed_on': obs.get('observed_on'),
                            'place_guess': obs.get('place_guess'),
                            'quality_grade': obs.get('quality_grade')
                        }
                        with open(f"{save_path}.json", 'w') as f:
                            json.dump(metadata, f, indent=2)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error saving: {e}")
                        
                # Stop if full
                if (hq_train + ch_train >= TRAIN_COUNT) and (hq_val + ch_val >= VAL_COUNT):
                    break
            
            # Force cleanup after batch (Memory Safety)
            import gc
            del images
            del batch_imgs
            gc.collect()
            
            if (hq_train + ch_train >= TRAIN_COUNT) and (hq_val + ch_val >= VAL_COUNT):
                break
                
        page += 1
        
    executor.shutdown(wait=False)
    pbar.close()
    
    final_train = hq_train + ch_train
    final_val = hq_val + ch_val
    success = (final_train >= TRAIN_COUNT and final_val >= VAL_COUNT)
    
    status = "✓" if success else "⚠️"
    print(f"  {status} {name}: {final_train}/{TRAIN_COUNT} train (HQ:{hq_train}/CH:{ch_train}), {final_val}/{VAL_COUNT} val")
    return success



def main():
    parser = argparse.ArgumentParser(description='Scrape iNaturalist with quality checks')
    parser.add_argument('--species', type=str, nargs='+', help='Specific species to scrape (folder name)')
    parser.add_argument('--resume', action='store_true', help='Resume from progress')
    parser.add_argument('--no-yolo', action='store_true', help='Skip YOLO validation')
    args = parser.parse_args()
    
    print("="*70)
    print("iNATURALIST SCRAPER V2 - WITH TIERED QUALITY")
    print("="*70)
    print(f"Target: {TRAIN_COUNT} train + {VAL_COUNT} val = {TOTAL_TARGET} per species")
    print(f"Quality Tiers:")
    print(f"  High Quality (85%): Blur>{HQ_BLUR_THRESHOLD}, Size>{HQ_MIN_SIZE}x{HQ_MIN_SIZE}")
    print(f"  Challenging  (15%): Blur>{CH_BLUR_THRESHOLD}, Size>{CH_MIN_SIZE}x{CH_MIN_SIZE}")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)
    
    # Load YOLO model
    yolo_model = None
    if YOLO_AVAILABLE and not args.no_yolo:
        print("Loading YOLO model...")
        yolo_model = YOLO('yolov8m.pt')
        print("  ✓ YOLO loaded")
    
    # Load species config
    species_map = load_species_config()
    print(f"Loaded {len(species_map)} species with taxon_ids")
    
    # Filter to specific species if requested
    if args.species:
        filtered_map = {}
        for s in args.species:
            if s in species_map:
                filtered_map[s] = species_map[s]
            else:
                print(f"❌ Species '{s}' not found")
        
        if not filtered_map:
            print("No valid species found to scrape.")
            return
        species_map = filtered_map
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"\nStarting scrape of {len(species_map)} species...")
    print()
    
    success_count = 0
    fail_count = 0
    
    for folder, info in species_map.items():
        try:
            if scrape_species(info, yolo_model):
                success_count += 1
            else:
                fail_count += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved.")
            break
        except Exception as e:
            print(f"  ❌ Error with {folder}: {e}")
            fail_count += 1
    
    print("\n" + "="*70)
    print("SCRAPE COMPLETE")
    print("="*70)
    print(f"Success: {success_count}/{len(species_map)}")
    print(f"Incomplete: {fail_count}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
