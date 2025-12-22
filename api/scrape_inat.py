import requests
import os
import time
import shutil

# --- CONFIGURATION ---
# The limit is set to 670 (approx 22k images total)
IMAGES_PER_ANIMAL = 670
OUTPUT_DIR = "training_data"

# Your Full List
ANIMALS = [
    # NATIVES
    "American Black Bear", "Grizzly Bear", "Moose", "White-tailed Deer",
    "American Bison", "Mountain Lion", "Coyote", "Bobcat", "Gray Wolf",
    "Raccoon", "North American Beaver", "Striped Skunk", "Virginia Opossum",
    "Eastern Gray Squirrel", "Red Fox", "Bald Eagle", "Red-tailed Hawk",
    "Great Blue Heron", "Wild Turkey", "Canada Goose", "American Alligator",
    "Eastern Box Turtle", "American Crocodile",
    # INVASIVES
    "Burmese Python", "Green Iguana", "Argentine Black and White Tegu",
    "Cane Toad", "European Starling", "House Sparrow", "Rock Pigeon",
    "Monk Parakeet", "Wild Boar", "Nutria"
]

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

def get_image_urls(taxon_id, target_count):
    """Fetch 'Research Grade' photo URLs with Pagination"""
    urls = []
    page = 1
    per_page = 200 # iNaturalist max per request
    
    print(f"    Fetching batches...", end='\r')

    while len(urls) < target_count:
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "per_page": per_page,
            "page": page,
            "photos": "true",
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
                    if photo_url not in urls:
                        urls.append(photo_url)
            
            if len(urls) >= target_count:
                break
            
            page += 1
            time.sleep(1) # Be respectful to API
            
        except Exception as e:
            print(f"    API Error: {e}")
            break
            
    return urls[:target_count]

def download_file(url, folder, index):
    try:
        response = requests.get(url, stream=True, timeout=10)
        ext = "jpg"
        filename = f"{index:03d}.{ext}"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        return True
    except: return False

if __name__ == "__main__":
    print(f"--- Starting Scrape for {len(ANIMALS)} Species (Target: {IMAGES_PER_ANIMAL}) ---")
    
    for animal in ANIMALS:
        print(f"\nProcessing: {animal}")
        taxon_id = get_taxon_id(animal)
        if not taxon_id:
            print(f"  [!] Could not find ID for {animal}")
            continue
            
        print(f"  > ID: {taxon_id}. Fetching URLs...")
        urls = get_image_urls(taxon_id, IMAGES_PER_ANIMAL)
        print(f"  > Found {len(urls)} photos.")
        
        safe_name = animal.lower().replace(" ", "_")
        split_idx = int(len(urls) * 0.8)
        
        for i, url in enumerate(urls):
            subset = 'train' if i < split_idx else 'val'
            target_folder = os.path.join(OUTPUT_DIR, subset, safe_name)
            os.makedirs(target_folder, exist_ok=True)
            
            download_file(url, target_folder, i)
            print(f"    Downloaded {i+1}/{len(urls)}", end='\r')

    print("\n\nDone! Data ready.")