"""
Fix broken taxon_ids in species_config.json
Then re-audit to verify all are correct
"""
import requests
import json
import time
from tqdm import tqdm

# Species that need fixing
SPECIES_TO_FIX = {
    # GENUS-LEVEL (need species-level)
    'Moose': 'Alces alces',
    'Striped Skunk': 'Mephitis mephitis',
    'Red Fox': 'Vulpes vulpes',
    'Northern Cardinal': 'Cardinalis cardinalis',
    'Axis Deer': 'Axis axis',
    
    # COMPLETELY WRONG
    'Gray Wolf': 'Canis lupus',
    'Wolverine': 'Gulo gulo',
    
    # MISSING
    'American Bison': 'Bison bison',
    'Green Iguana': 'Iguana iguana',
}

print("="*70)
print("STEP 1: Finding correct taxon_ids")
print("="*70)

correct_ids = {}

for common_name, scientific_name in SPECIES_TO_FIX.items():
    print(f"\n{common_name} ({scientific_name})...")
    
    try:
        r = requests.get(
            'https://api.inaturalist.org/v1/taxa',
            params={'q': scientific_name, 'rank': 'species'},
            timeout=15
        )
        
        if r.status_code == 200:
            results = r.json().get('results', [])
            for result in results:
                if result.get('rank') == 'species':
                    iconic = result.get('iconic_taxon_name', '')
                    # Make sure it's an animal
                    if iconic in ['Mammalia', 'Aves', 'Reptilia', 'Amphibia']:
                        taxon_id = result['id']
                        inat_name = result['name']
                        print(f"  ✓ {inat_name} (ID: {taxon_id}, Type: {iconic})")
                        correct_ids[common_name] = taxon_id
                        break
            else:
                print(f"  ❌ No animal species found")
        else:
            print(f"  ❌ HTTP {r.status_code}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    time.sleep(0.5)

print("\n" + "="*70)
print("STEP 2: Updating species_config.json")
print("="*70)

with open('species_config.json', 'r') as f:
    config = json.load(f)

updated_count = 0

for entry in config:
    name = entry['name']
    if name in correct_ids:
        if 'taxonomy' not in entry:
            entry['taxonomy'] = {}
        old_id = entry['taxonomy'].get('taxon_id', 'MISSING')
        new_id = correct_ids[name]
        entry['taxonomy']['taxon_id'] = new_id
        print(f"  Updated: {name} ({old_id} -> {new_id})")
        updated_count += 1

# Save
with open('species_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print(f"\n✓ Updated {updated_count} species")

print("\n" + "="*70)
print("STEP 3: Re-auditing (with longer delays to avoid rate limit)")
print("="*70)

# Re-audit with longer delays
issues = []
verified = []

VALID_TAXONS = ['Mammalia', 'Aves', 'Reptilia', 'Amphibia']

for entry in tqdm(config, desc="Verifying"):
    name = entry['name']
    taxon_id = entry.get('taxonomy', {}).get('taxon_id')
    
    if not taxon_id:
        issues.append({'name': name, 'problem': 'MISSING ID'})
        continue
    
    try:
        r = requests.get(f'https://api.inaturalist.org/v1/taxa/{taxon_id}', timeout=15)
        
        if r.status_code == 200:
            data = r.json().get('results', [{}])[0]
            iconic = data.get('iconic_taxon_name', '')
            rank = data.get('rank', '')
            inat_name = data.get('name', '')
            inat_common = data.get('preferred_common_name', '')
            
            is_animal = iconic in VALID_TAXONS
            is_species = rank in ['species', 'subspecies']
            
            if is_animal and is_species:
                verified.append(name)
            else:
                issues.append({
                    'name': name,
                    'taxon_id': taxon_id,
                    'inat_name': inat_name,
                    'iconic': iconic,
                    'rank': rank,
                    'problem': 'NOT ANIMAL' if not is_animal else f'Bad rank: {rank}'
                })
        elif r.status_code == 429:
            issues.append({'name': name, 'taxon_id': taxon_id, 'problem': 'Rate limited - retry later'})
        else:
            issues.append({'name': name, 'taxon_id': taxon_id, 'problem': f'HTTP {r.status_code}'})
            
    except Exception as e:
        issues.append({'name': name, 'problem': str(e)[:30]})
    
    time.sleep(0.5)  # Longer delay

# Report
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"✓ Verified: {len(verified)}/100")
print(f"❌ Issues: {len(issues)}")

if issues:
    print("\nREMAINING ISSUES:")
    for issue in issues:
        print(f"  - {issue['name']}: {issue['problem']}")

# Save report
with open('taxon_fix_report.json', 'w') as f:
    json.dump({'verified': len(verified), 'issues': issues}, f, indent=2)

print(f"\n📁 Final report saved to: taxon_fix_report.json")
