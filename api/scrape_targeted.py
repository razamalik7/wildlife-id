"""
Targeted Scraping Script
========================
Scrape additional images ONLY for classes that need improvement.

Reads the confusion report from analyze_confusion.py and automatically
scrapes more diverse images for the most confused classes.

Usage:
  1. First run analyze_confusion.py to generate the report
  2. Then run this script:
     
     python scrape_targeted.py --report grandmaster_b3_final_confusion_report.json --count 500
     
  This will scrape 500 additional images per confused class.
"""

import os
import json
import time
import argparse
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available - skipping animal validation")
    YOLO_AVAILABLE = False

# iNaturalist API config
INAT_API_BASE = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "WildlifeIDApp/1.0"}

# Species name to iNaturalist taxon mapping (from your existing scraper)
SPECIES_TO_TAXON = {
    'american_alligator': 'Alligator mississippiensis',
    'american_badger': 'Taxidea taxus',
    'american_bison': 'Bison bison',
    'american_black_bear': 'Ursus americanus',
    'american_bullfrog': 'Lithobates catesbeianus',
    'american_crocodile': 'Crocodylus acutus',
    'american_flamingo': 'Phoenicopterus ruber',
    'american_marten': 'Martes americana',
    'american_mink': 'Neovison vison',
    'american_red_squirrel': 'Tamiasciurus hudsonicus',
    'arctic_fox': 'Vulpes lagopus',
    'argentine_black_and_white_tegu': 'Salvator merianae',
    'axis_deer': 'Axis axis',
    'bald_eagle': 'Haliaeetus leucocephalus',
    'barn_owl': 'Tyto alba',
    'bighorn_sheep': 'Ovis canadensis',
    'black-footed_ferret': 'Mustela nigripes',
    'black-tailed_prairie_dog': 'Cynomys ludovicianus',
    'blue_jay': 'Cyanocitta cristata',
    'bobcat': 'Lynx rufus',
    'brown_anole': 'Anolis sagrei',
    'brown_pelican': 'Pelecanus occidentalis',
    'burmese_python': 'Python bivittatus',
    'california_sea_lion': 'Zalophus californianus',
    'canada_goose': 'Branta canadensis',
    'cane_toad': 'Rhinella marina',
    'caribou': 'Rangifer tarandus',
    'common_garter_snake': 'Thamnophis sirtalis',
    'common_snapping_turtle': 'Chelydra serpentina',
    'coyote': 'Canis latrans',
    'thinhorn_sheep': 'Ovis dalli',
    'desert_tortoise': 'Gopherus agassizii',
    'domestic_cat': 'Felis catus',
    'domestic_dog': 'Canis lupus familiaris',
    'common_box_turtle': 'Terrapene carolina',
    'eastern_chipmunk': 'Tamias striatus',
    'eastern_copperhead': 'Agkistrodon contortrix',
    'eastern_cottontail': 'Sylvilagus floridanus',
    'eastern_fox_squirrel': 'Sciurus niger',
    'eastern_gray_squirrel': 'Sciurus carolinensis',
    'eastern_hellbender': 'Cryptobranchus alleganiensis',
    'elk': 'Cervus canadensis',
    'european_starling': 'Sturnus vulgaris',
    'gemsbok': 'Oryx gazella',
    'gila_monster': 'Heloderma suspectum',
    'gray_fox': 'Urocyon cinereoargenteus',
    'gray_wolf': 'Canis lupus',
    'great_blue_heron': 'Ardea herodias',
    'great_horned_owl': 'Bubo virginianus',
    'green_anole': 'Anolis carolinensis',
    'green_iguana': 'Iguana iguana',
    'grizzly_bear': 'Ursus arctos horribilis',
    'groundhog': 'Marmota monax',
    'harbor_seal': 'Phoca vitulina',
    'house_sparrow': 'Passer domesticus',
    'jaguar': 'Panthera onca',
    'mallard': 'Anas platyrhynchos',
    'monk_parakeet': 'Myiopsitta monachus',
    'moose': 'Alces alces',
    'mountain_goat': 'Oreamnos americanus',
    'mountain_lion': 'Puma concolor',
    'muskox': 'Ovibos moschatus',
    'mute_swan': 'Cygnus olor',
    'nile_monitor': 'Varanus niloticus',
    'nine-banded_armadillo': 'Dasypus novemcinctus',
    'north_american_beaver': 'Castor canadensis',
    'north_american_porcupine': 'Erethizon dorsatum',
    'north_american_river_otter': 'Lontra canadensis',
    'northern_cardinal': 'Cardinalis cardinalis',
    'northern_elephant_seal': 'Mirounga angustirostris',
    'nutria': 'Myocastor coypus',
    'ocelot': 'Leopardus pardalis',
    'osprey': 'Pandion haliaetus',
    'painted_turtle': 'Chrysemys picta',
    'peregrine_falcon': 'Falco peregrinus',
    'polar_bear': 'Ursus maritimus',
    'pronghorn': 'Antilocapra americana',
    'raccoon': 'Procyon lotor',
    'red_fox': 'Vulpes vulpes',
    'eastern_newt': 'Notophthalmus viridescens',
    'red-tailed_hawk': 'Buteo jamaicensis',
    'rhesus_macaque': 'Macaca mulatta',
    'ring-necked_pheasant': 'Phasianus colchicus',
    'rock_pigeon': 'Columba livia',
    'sea_otter': 'Enhydra lutris',
    'snowshoe_hare': 'Lepus americanus',
    'snowy_owl': 'Bubo scandiacus',
    'spotted_salamander': 'Ambystoma maculatum',
    'striped_skunk': 'Mephitis mephitis',
    'tokay_gecko': 'Gekko gecko',
    'veiled_chameleon': 'Chamaeleo calyptratus',
    'virginia_opossum': 'Didelphis virginiana',
    'walrus': 'Odobenus rosmarus',
    'western_diamondback_rattlesnake': 'Crotalus atrox',
    'white-tailed_deer': 'Odocoileus virginianus',
    'whooping_crane': 'Grus americana',
    'wild_boar': 'Sus scrofa',
    'wild_turkey': 'Meleagris gallopavo',
    'wolverine': 'Gulo gulo',
    'yellow-bellied_marmot': 'Marmota flaviventris',
}


def get_taxon_id(scientific_name):
    """Get iNaturalist taxon ID from scientific name."""
    url = f"{INAT_API_BASE}/taxa"
    params = {"q": scientific_name, "per_page": 1}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            if results:
                return results[0]['id']
    except Exception as e:
        print(f"Error getting taxon ID for {scientific_name}: {e}")
    return None


def get_observations(taxon_id, per_page=200, page=1, quality_grade='research'):
    """Get observations with photos from iNaturalist."""
    url = f"{INAT_API_BASE}/observations"
    params = {
        "taxon_id": taxon_id,
        "photos": True,
        "quality_grade": quality_grade,
        "per_page": per_page,
        "page": page,
        "order_by": "random"  # Get random samples for diversity
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json().get('results', [])
    except Exception as e:
        print(f"Error getting observations: {e}")
    return []


# ============== FILTER FUNCTIONS ==============

def calculate_blur_score(image):
    """Calculate blur score using Laplacian variance (higher = sharper)"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_dead_animal(observation):
    """Check if observation mentions dead animal"""
    dead_keywords = ['dead', 'roadkill', 'deceased', 'carcass', 'deceased', 'mortality', 'remains', 'skull', 'bone', 'skeleton']
    description = (observation.get('description') or '').lower()
    for field in observation.get('ofvs', []):
        if field.get('value', '').lower() in dead_keywords: return True
    for keyword in dead_keywords:
        if keyword in description: return True
    return False

def process_image_quality(image, yolo_model=None):
    """
    Check image quality and return (final_img, is_valid, reason)
    Checks: Size, Blur, YOLO animal presence
    """
    # 1. Size Check (Min 100 on smallest side)
    if min(image.size) < 100:
        return None, False, "too_small"
        
    # 2. Blur Check (Min 80 variance)
    blur = calculate_blur_score(image)
    if blur < 80:
        return None, False, f"blurry_{blur:.0f}"
        
    # 3. YOLO Check (If available)
    if yolo_model:
        # Animal classes in COCO (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
        # Note: YOLO is imperfect for exotic species, so we use a low confidence threshold
        # and checking for ANY object is often safer than just animals for very exotic things
        # But for 'wildlife' usually it picks up 'bird' or 'animal'
        try:
             img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
             results = yolo_model(img_cv, verbose=False, conf=0.1)
             if not results or len(results[0].boxes) == 0:
                 return None, False, "no_detection"
        except:
             pass # Skip check if error
             
    return image, True, "ok"

def download_image_buffer(url):
    """Download image to memory buffer."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert('RGB')
    except:
        pass
    return None

def download_image(url, save_path):
    """Legacy wrapper for consistency"""
    # This is replaced by the filtered logic in valid path
    pass


def get_existing_observation_ids(class_dir):
    """
    Scan existing JSON files to extract observation IDs and URLs we already have.
    This prevents re-downloading the same images.
    """
    existing_ids = set()
    existing_urls = set()
    
    if not os.path.exists(class_dir):
        return existing_ids, existing_urls
    
    for filename in os.listdir(class_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(class_dir, filename)
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    # Extract observation ID from JSON content (Preferred)
                    obs_id = meta.get('observation_id') or meta.get('id')
                    if obs_id:
                        existing_ids.add(int(obs_id))
                        
                    # Extract URL
                    url = meta.get('url', '')
                    if url:
                        existing_urls.add(url)
                        
                    # Also check filename for observation ID pattern (Legacy/Fallback)
                    parts = filename.replace('.json', '').split('_')
                    for part in parts:
                        if part.isdigit() and len(part) > 5:
                            existing_ids.add(int(part))
            except:
                pass
    
    return existing_ids, existing_urls


def scrape_class(class_name, target_count, output_dir):
    """Scrape additional images for a single class, avoiding duplicates."""
    scientific_name = SPECIES_TO_TAXON.get(class_name)
    if not scientific_name:
        print(f"  ⚠️ No mapping for {class_name}")
        return 0
    
    taxon_id = get_taxon_id(scientific_name)
    if not taxon_id:
        print(f"  ⚠️ Could not find taxon ID for {scientific_name}")
        return 0
    
    class_dir = os.path.join(output_dir, 'train', class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Get existing observation IDs and URLs to avoid duplicates
    existing_ids, existing_urls = get_existing_observation_ids(class_dir)
    
    # CHECK GLOBAL DATASET (training_data_v2)
    # We assume 'training_data_v2' is a sibling of 'training_data_refined'
    api_root = os.path.dirname(os.path.abspath(output_dir))
    
    # Check both TRAIN and VAL splits in V2
    v2_train_dir = os.path.join(api_root, 'training_data_v2', 'train', class_name)
    v2_val_dir = os.path.join(api_root, 'training_data_v2', 'val', class_name)
    
    total_v2_found = 0
    
    for v2_dir in [v2_train_dir, v2_val_dir]:
        if os.path.exists(v2_dir):
            vids, vurls = get_existing_observation_ids(v2_dir)
            existing_ids.update(vids)
            existing_urls.update(vurls)
            total_v2_found += len(vids)
        else:
             print(f"  Dedup: Checked {v2_dir} (Not Found)")
             
    print(f"  Dedup: Checked V2 dataset, found {total_v2_found} existing items")
        
    existing_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  📂 {class_name}: {existing_count} local images, {len(existing_ids)} total tracked IDs")
    
    downloaded = 0
    skipped_duplicates = 0
    page = 1
    max_pages = 30  # Increased to find more unique images
    
    # Deduplication: Check V2 (Historical) AND Refined (Current Output)
    # The 'existing_ids' set covers both to prevent duplicates.
    # BUT, we only count the 'Refined' images towards our 'target_count' progress.
    
    local_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    local_count = len(local_files)
    
    remaining_needed = target_count - local_count
    
    if remaining_needed <= 0:
        print(f"  ✅ Class {class_name} complete: {local_count}/{target_count} images ready.")
        return 0
        
    print(f"  📉 Status: Have {local_count}, Need {remaining_needed} more.")

    # Load YOLO locally for this process
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO('yolov8m.pt') 
        except:
             pass

    # Threaded Download Setup 
    # Use more threads for pure downloading since CPU isn't blocked by YOLO here
    # INCREASED to 32 to fix GPU Starvation (Network Bottleneck)
    executor = ThreadPoolExecutor(max_workers=32) 
    
    pbar = tqdm(total=remaining_needed, desc=f"  {class_name}", leave=False)
    
    BATCH_SIZE = 32 # GPU Batch Size
    
    BATCH_SIZE = 32 # GPU Batch Size
    
    # NEW: Function to run inside thread (Download + CPU Checks)
    def download_and_verify(url):
        img = download_image_buffer(url)
        if img is None: return None, "download_fail"
        
        # CPU CHECKS (Parallelized)
        if min(img.size) < 100:
            return None, "too_small"
            
        blur = calculate_blur_score(img)
        if blur < 80:
            return None, f"blurry_{blur:.0f}"
            
        return img, "ok"

    while downloaded < remaining_needed and page <= max_pages:
        observations = get_observations(taxon_id, per_page=200, page=page)
        if not observations:
            break
            
        # 1. Filter Candidates (Deduplication + Dead + Metadata)
        candidates = []
        candidate_obs = []
        
        for obs in observations:
            if downloaded >= remaining_needed: break  # FIX: Use remaining_needed, not target_count
            
            obs_id = obs.get('id')
            if obs_id in existing_ids:
                skipped_duplicates += 1
                continue
            
            if is_dead_animal(obs): continue
                
            photos = obs.get('photos', [])
            if not photos: continue
            
            photo = photos[0]
            photo_url = photo.get('url', '').replace('square', 'medium')
            if not photo_url: continue
            
            if photo_url in existing_urls:
                skipped_duplicates += 1
                continue
                
            candidates.append(photo_url)
            candidate_obs.append(obs)
            existing_ids.add(obs_id) # Dedup immediately
            existing_urls.add(photo_url)
            
        if not candidates:
            page += 1
            continue
            
        # 2. Process in GPU Batches
        num_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(num_batches):
            if downloaded >= remaining_needed: break
            
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            batch_urls = candidates[start_idx:end_idx]
            batch_obs_data = candidate_obs[start_idx:end_idx]
            
            # A. Parallel Download AND Verification (Offload CPU work)
            # Use 'map' to keep order aligned with batch_obs_data
            results = list(executor.map(download_and_verify, batch_urls))
            
            valid_images = []
            valid_obs = []
            valid_urls = []
            
            # B. Collect Valid Results (Main thread just filters lists)
            for j, (img, status) in enumerate(results):
                if img is None: continue # rejected by thread
                
                valid_images.append(img)
                valid_obs.append(batch_obs_data[j])
                valid_urls.append(batch_urls[j])
                
            if not valid_images: continue
            
            # C. Batch YOLO (GPU)
            final_indices = []
            if yolo_model:
                try:
                    # Convert to cv2 BGR (Fast enough on main thread for small batch)
                    imgs_cv = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in valid_images]
                    
                    results = yolo_model(imgs_cv, verbose=False, conf=0.1, batch=len(imgs_cv))
                    
                    for k, res in enumerate(results):
                        if res.boxes and len(res.boxes) > 0:
                            final_indices.append(k)
                except Exception as e:
                    final_indices = range(len(valid_images))
            else:
                 final_indices = range(len(valid_images))
                 
            # D. Save Results
            for k in final_indices:
                if downloaded >= remaining_needed: break
                
                img = valid_images[k]
                obs_data = valid_obs[k]
                url = valid_urls[k]
                
                obs_id = obs_data.get('id')
                save_name = f"targeted_{obs_id}.jpg"
                save_path = os.path.join(class_dir, save_name)
                
                try:
                    img.save(save_path, quality=95)
                    
                    location = obs_data.get('location', '')
                    lat_str, lng_str = (location.split(',') + ['', ''])[:2] if location else ('', '')
                    meta = {
                        'observation_id': obs_id,
                        'lat': float(lat_str) if lat_str.strip() else None,
                        'lng': float(lng_str) if lng_str.strip() else None,
                        'date': obs_data.get('observed_on', ''),
                        'url': url,
                        'source': 'targeted_scrape_batch',
                        'quality': 'checked'
                    }
                    
                    json_path = save_path.replace('.jpg', '.json')
                    with open(json_path, 'w') as f:
                        json.dump(meta, f)
                        
                    downloaded += 1
                    pbar.update(1)
                except:
                    pass
            
            del valid_images, results
            
        page += 1
        
    executor.shutdown(wait=False)
    pbar.close()
    
    if skipped_duplicates > 0:
        print(f"  ℹ️ Skipped {skipped_duplicates} duplicates")
    
    return downloaded


def scrape_from_report(report_path, count_per_class=500, top_n=10):
    """
    Read confusion report and scrape more images for confused classes.
    """
    print(f"🎯 TARGETED SCRAPING FROM CONFUSION REPORT")
    print(f"   Report: {report_path}")
    print(f"   Images per class: {count_per_class}")
    print(f"   Top N classes: {top_n}")
    
    # Load confusion report
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    classes_to_scrape = report.get('classes_needing_more_data', [])[:top_n]
    
    if not classes_to_scrape:
        print("❌ No classes found in report")
        return
    
    print(f"\n📋 Classes to scrape:")
    for cls in classes_to_scrape:
        print(f"   - {cls}")
    
    print(f"\n🌐 Starting scrape...")
    
    total_downloaded = 0
    for class_name in classes_to_scrape:
        print(f"\n📷 Scraping {class_name}...")
        downloaded = scrape_class(class_name, count_per_class, './training_data_cropped')
        total_downloaded += downloaded
        print(f"   ✅ Downloaded {downloaded} new images")
        time.sleep(1)  # Rate limiting between classes
    
    print(f"\n🏆 SCRAPING COMPLETE!")
    print(f"   Total new images: {total_downloaded}")
    print(f"\n📋 Next steps:")
    print(f"   1. Run fine-tuning with the augmented dataset:")
    print(f"      python targeted_finetune.py --model <your_model.pth> --type <b3|convnext>")


def main():
    parser = argparse.ArgumentParser(description='Targeted Scraping for Confused Classes')
    parser.add_argument('--report', type=str, required=True, help='Path to confusion report JSON')
    parser.add_argument('--count', type=int, default=500, help='Images to scrape per class')
    parser.add_argument('--top', type=int, default=10, help='Number of top confused classes to scrape')
    args = parser.parse_args()
    
    scrape_from_report(args.report, args.count, args.top)


if __name__ == "__main__":
    main()
