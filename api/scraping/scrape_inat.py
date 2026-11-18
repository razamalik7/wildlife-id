import requests
import os
import time
import shutil
import json

# --- CONFIGURATION ---
# The limit is set to 670 (approx 22k images total)
IMAGES_PER_ANIMAL = 670
OUTPUT_DIR = "training_data"
CONFIG_FILE = "species_config.json"

def load_animals_from_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            return [entry['name'] for entry in data]
    return []

ANIMALS = load_animals_from_config()

def get_taxon_id(animal_name):
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": animal_name, "rank": "species,subspecies", "per_page": 1}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results: return results[0]['id']
    except:
        pass
    return None

def get_image_data(taxon_id, target_count):
    """Fetch 'Research Grade' photo URLs + Metadata with Pagination"""
    data_points = []
    page = 1
    per_page = 200
    
    print(f"    Fetching batches...", end='\r')

    while len(data_points) < target_count:
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "per_page": per_page,
            "page": page,
            "photos": "true",
            "geo": "true", # Ensure location exists
            "license": "cc-by,cc-by-nc"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                break
                
            results = response.json().get('results', [])
            if not results:
                break 
            
            for obs in results:
                if len(obs['photos']) > 0:
                    photo_url = obs['photos'][0]['url'].replace("square", "medium")
                    
                    # Extract Metadata
                    lat = None
                    lng = None
                    if obs.get('geojson'):
                        lng, lat = obs['geojson']['coordinates']
                    elif obs.get('location'):
                        # formatted as "lat,lng" sometimes
                        parts = obs['location'].split(',')
                        if len(parts) == 2:
                            lat, lng = float(parts[0]), float(parts[1])
                            
                    date_str = obs.get('observed_on', None) # YYYY-MM-DD
                    
                    # Deduplication check
                    if not any(d['url'] == photo_url for d in data_points):
                        data_points.append({
                            "url": photo_url,
                            "lat": lat,
                            "lng": lng,
                            "date": date_str
                        })
            
            if len(data_points) >= target_count:
                break
            
            page += 1
            time.sleep(1)
            
        except Exception as e:
            print(f"    API Error: {e}")
            break
            
    return data_points[:target_count]

def download_file(item, folder, index):
    try:
        # 1. Download Image
        response = requests.get(item['url'], stream=True, timeout=10)
        ext = "jpg"
        filename = f"{index:03d}.{ext}"
        filepath = os.path.join(folder, filename)
        
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
            
        # 2. Save Metadata Sidecar
        json_filename = f"{index:03d}.json"
        json_filepath = os.path.join(folder, json_filename)
        with open(json_filepath, 'w') as f:
            json.dump({
                "lat": item['lat'],
                "lng": item['lng'],
                "date": item['date'],
                "url": item['url']
            }, f)
            
        return True
    except: return False

if __name__ == "__main__":
    print(f"--- Starting Scrape (Metadata Enabled) for {len(ANIMALS)} Species (Target: {IMAGES_PER_ANIMAL}) ---")
    
    for animal in ANIMALS:
        print(f"\nProcessing: {animal}")
        taxon_id = get_taxon_id(animal)
        if not taxon_id:
            print(f"  [!] Could not find ID for {animal}")
            continue

        # IDEMPOTENCY CHECK (Metadata Aware)
        safe_name = animal.lower().replace(" ", "_")
        train_path = os.path.join(OUTPUT_DIR, 'train', safe_name)
        val_path = os.path.join(OUTPUT_DIR, 'val', safe_name)
        
        needed_count = int(IMAGES_PER_ANIMAL * 0.8 * 0.7) 
        
        has_metadata = False
        if os.path.exists(train_path):
            # Count JSON sidecars to ensure we have metadata
            json_count = len([f for f in os.listdir(train_path) if f.endswith('.json')])
            if json_count > needed_count:
                has_metadata = True
                
        if has_metadata:
             print(f"  [i] Skipping {animal} (Found images + metadata)")
             continue
        elif os.path.exists(train_path):
             print(f"  [REFRESH] {animal}: Old data found without metadata. Deleting to re-scrape...")
             if os.path.exists(train_path): shutil.rmtree(train_path)
             if os.path.exists(val_path): shutil.rmtree(val_path)
            
        print(f"  > ID: {taxon_id}. Fetching Data...")
        items = get_image_data(taxon_id, IMAGES_PER_ANIMAL)
        print(f"  > Found {len(items)} observations.")
        
        safe_name = animal.lower().replace(" ", "_")
        split_idx = int(len(items) * 0.8)
        
        for i, item in enumerate(items):
            subset = 'train' if i < split_idx else 'val'
            target_folder = os.path.join(OUTPUT_DIR, subset, safe_name)
            os.makedirs(target_folder, exist_ok=True)
            
            download_file(item, target_folder, i)
            print(f"    Downloaded {i+1}/{len(items)}", end='\r')

    print("\n\nDone! Data ready.")