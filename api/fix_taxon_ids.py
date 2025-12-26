"""
Fix taxon_ids that point to genus instead of species
"""
import requests
import json
import time

# Species that have wrong taxon_ids (genus instead of species)
FIXES_NEEDED = {
    'Moose': 'Alces alces',
    'Striped Skunk': 'Mephitis mephitis', 
    'Red Fox': 'Vulpes vulpes',
    'Northern Cardinal': 'Cardinalis cardinalis',
    'Axis Deer': 'Axis axis',
    'American Bison': 'Bison bison',
    'Green Iguana': 'Iguana iguana'
}

print("="*70)
print("FIXING GENUS-LEVEL TAXON IDS")
print("="*70)

correct_ids = {}

for common_name, scientific_name in FIXES_NEEDED.items():
    print(f"\n{common_name} ({scientific_name})...")
    
    r = requests.get(
        'https://api.inaturalist.org/v1/taxa',
        params={
            'q': scientific_name,
            'rank': 'species'
        },
        timeout=10
    )
    
    if r.status_code == 200:
        results = r.json().get('results', [])
        for result in results:
            if result.get('rank') == 'species':
                taxon_id = result['id']
                inat_name = result['name']
                iconic = result.get('iconic_taxon_name', '')
                print(f"  ✓ Found: {inat_name} (ID: {taxon_id}, Type: {iconic})")
                correct_ids[common_name] = {
                    'taxon_id': taxon_id,
                    'scientific': inat_name
                }
                break
    else:
        print(f"  ❌ HTTP {r.status_code}")
    
    time.sleep(0.3)

print("\n" + "="*70)
print("CORRECT TAXON IDS:")
print("="*70)
for name, data in correct_ids.items():
    print(f"  {name}: {data['taxon_id']} ({data['scientific']})")

# Save for reference
with open('taxon_fixes.json', 'w') as f:
    json.dump(correct_ids, f, indent=2)

print(f"\n📁 Saved to: taxon_fixes.json")
print("\nNow update species_config.json with these correct IDs!")
