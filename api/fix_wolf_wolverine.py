import json
import requests
import time

# Find correct IDs for Gray Wolf and Wolverine
to_find = {
    'Gray Wolf': 'Canis lupus',  # NOT Canis familiaris (dog)
    'Wolverine': 'Gulo gulo'      # NOT Mergus merganser (duck)
}

print("Finding CORRECT taxon_ids:")
correct = {}

for name, sci_name in to_find.items():
    print(f"\n{name} ({sci_name})...")
    
    # Search with exact match
    r = requests.get(
        'https://api.inaturalist.org/v1/taxa',
        params={
            'q': sci_name,
            'rank': 'species',
            'per_page': 20
        },
        timeout=15
    )
    
    if r.status_code == 200:
        results = r.json().get('results', [])
        print(f"  Found {len(results)} results")
        
        for result in results:
            inat_name = result.get('name', '')
            iconic = result.get('iconic_taxon_name', '')
            
            # For Canis lupus, make sure it's NOT familiaris (dog)
            if sci_name == 'Canis lupus' and 'lupus' in inat_name:
                print(f"  ✓ FOUND: {inat_name} (ID: {result['id']}, Type: {iconic})")
                correct[name] = result['id']
                break
            # For Gulo gulo, make sure it's a mammal
            elif sci_name == 'Gulo gulo' and iconic == 'Mammalia' and 'Gulo' in inat_name:
                print(f"  ✓ FOUND: {inat_name} (ID: {result['id']}, Type: {iconic})")
                correct[name] = result['id']
                break
        else:
            # Print what we found
            for r in results[:5]:
                print(f"    - {r['name']} (ID: {r['id']}, Type: {r.get('iconic_taxon_name', '')})")
    else:
        print(f"  HTTP {r.status_code}")
    
    time.sleep(0.5)

# Now update the config
print("\n" + "="*70)

if correct:
    print(f"Updating species_config.json with:")
    for name, tid in correct.items():
        print(f"  {name}: {tid}")
    
    config = json.load(open('species_config.json'))
    
    for entry in config:
        if entry['name'] in correct:
            old_id = entry.get('taxonomy', {}).get('taxon_id')
            entry['taxonomy']['taxon_id'] = correct[entry['name']]
            print(f"\n  Updated: {entry['name']} ({old_id} -> {correct[entry['name']]})")
    
    with open('species_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n✓ Config updated!")
else:
    print("No corrections found - need manual lookup")
