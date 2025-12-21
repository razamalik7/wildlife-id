import requests
import os
import time
import shutil

# --- CONFIGURATION ---
# 1. THE LIST: Add all 30 of your animals here exactly as they appear in your DB
ANIMALS = [
    # --- NATIVES ---
    "American Black Bear",
    "Grizzly Bear",
    "Moose",
    "White-tailed Deer",
    "American Bison",
    "Mountain Lion",
    "Coyote",
    "Bobcat",
    "Gray Wolf",
    "Raccoon",
    "North American Beaver",
    "Striped Skunk",
    "Virginia Opossum",
    "Eastern Gray Squirrel",
    "Red Fox",
    "Bald Eagle",
    "Red-tailed Hawk",
    "Great Blue Heron",
    "Wild Turkey",
    "Canada Goose",
    "American Alligator",
    "Eastern Box Turtle",
    "American Crocodile",

    # --- INVASIVES ---
    "Burmese Python",
    "Green Iguana",
    "Argentine Black and White Tegu",
    "Cane Toad",
    "European Starling",
    "House Sparrow",
    "Rock Pigeon",
    "Monk Parakeet",
    "Wild Boar",
    "Nutria"
]

IMAGES_PER_ANIMAL = 670 
OUTPUT_DIR = "training_data" 

def get_taxon_id(animal_name):
    """Ask iNaturalist for the ID (e.g., 'Bear' -> 41638)"""
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": animal_name, "rank": "species,subspecies", "per_page": 1}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()['results']
        if results: return results[0]['id']
    return None

def get_image_urls(taxon_id, limit):
    """Get verified 'Research Grade' photo URLs"""
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research", 
        "per_page": limit,
        "photos": "true",
        "license": "cc-by,cc-by-nc" 
    }
    response = requests.get(url, params=params)
    urls = []
    if response.status_code == 200:
        for obs in response.json()['results']:
            if len(obs['photos']) > 0:
                urls.append(obs['photos'][0]['url'].replace("square", "medium"))
    return urls

def download_file(url, folder, index):
    try:
        response = requests.get(url, stream=True, timeout=10)
        ext = "jpg" # Force jpg extension
        filename = f"{index:03d}.{ext}"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        return True
    except: return False

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"--- Starting Scrape for {len(ANIMALS)} Species ---")
    
    for animal in ANIMALS:
        print(f"\nProcessing: {animal}")
        taxon_id = get_taxon_id(animal)
        if not taxon_id:
            print(f"  [!] Could not find ID for {animal}")
            continue
            
        print(f"  > ID: {taxon_id}. Fetching URLs...")
        urls = get_image_urls(taxon_id, IMAGES_PER_ANIMAL)
        
        # Safe folder name (e.g. "red_fox")
        safe_name = animal.lower().replace(" ", "_")
        
        # Split: 80% Train, 20% Validation
        split_idx = int(len(urls) * 0.8)
        
        for i, url in enumerate(urls):
            subset = 'train' if i < split_idx else 'val'
            target_folder = os.path.join(OUTPUT_DIR, subset, safe_name)
            os.makedirs(target_folder, exist_ok=True)
            
            download_file(url, target_folder, i)
            print(f"    Downloaded {i+1}/{len(urls)}", end='\r')

    print("\n\nDone! Data ready.")