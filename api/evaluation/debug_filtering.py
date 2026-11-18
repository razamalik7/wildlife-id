"""
Debug version of scraper to see what's filtering images
"""
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import json

# Load moose config
with open('species_config.json') as f:
    config = json.load(f)
moose = [e for e in config if 'moose' in e['name'].lower()][0]
taxon_id = moose['taxonomy']['taxon_id']

print(f"Testing moose (taxon_id: {taxon_id})")

# Get one observation
r = requests.get(
    'https://api.inaturalist.org/v1/observations',
    params={
        'taxon_id': taxon_id,
        'quality_grade': 'research',
        'photos': 'true',
        'per_page': 10
    }
)

results = r.json().get('results', [])
print(f"\nGot {len(results)} observations\n")

def calculate_blur_score(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

tested = 0
for obs in results[:5]:
    tested += 1
    photo_url = obs['photos'][0]['url'].replace('square', 'medium')
    
    print(f"\n--- Observation {obs['id']} ---")
    print(f"URL: {photo_url}")
    
    try:
        response = requests.get(photo_url, timeout=15)
        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code}")
            continue
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"  Original size: {image.size}")
        
        if min(image.size) < 100:
            print(f"  ❌ Too small (< 100)")
            continue
        
        blur_score = calculate_blur_score(image)
        print(f"  Blur score: {blur_score:.1f}")
        
        # Check thresholds
        if blur_score >= 200 and min(image.size) >= 100:
            print(f"  ✅ HIGH QUALITY")
        elif blur_score >= 80 and min(image.size) >= 50:
            print(f"  ✅ CHALLENGING")
        else:
            print(f"  ❌ REJECTED (blur too low or size too small)")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\nTested {tested} observations")
