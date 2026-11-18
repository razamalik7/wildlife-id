import requests
import json

# Find Phoenicopterus ruber taxon_id
print("Finding Phoenicopterus ruber...")
r = requests.get('https://api.inaturalist.org/v1/taxa', 
                 params={'q': 'Phoenicopterus ruber', 'rank': 'species'})
results = r.json()['results']

for x in results[:5]:
    print(f"  {x['id']}: {x['name']} ({x.get('preferred_common_name','')}) - {x.get('iconic_taxon_name','')}")

# Get correct ID
correct_id = None
for x in results:
    if 'ruber' in x['name'] and x.get('iconic_taxon_name') == 'Aves':
        correct_id = x['id']
        print(f"\n✓ Using: {x['name']} (ID: {correct_id})")
        break

if correct_id:
    # Update species_config.json
    config = json.load(open('species_config.json'))
    
    for entry in config:
        if entry['name'] == 'American Flamingo':
            old_id = entry['taxonomy'].get('taxon_id')
            entry['taxonomy']['taxon_id'] = correct_id
            print(f"\n  Updated American Flamingo: {old_id} -> {correct_id}")
            break
    
    with open('species_config.json', 'w') as f:
        json.dump(config, f, indent=4)
