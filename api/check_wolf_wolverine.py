import json
import requests

# Check Gray Wolf and Wolverine specifically
config = json.load(open('species_config.json'))

for entry in config:
    if entry['name'] in ['Gray Wolf', 'Wolverine']:
        tid = entry.get('taxonomy', {}).get('taxon_id')
        print(f"\n{entry['name']} (taxon_id: {tid})")
        
        # Verify against iNat
        r = requests.get(f'https://api.inaturalist.org/v1/taxa/{tid}', timeout=10)
        if r.status_code == 200:
            data = r.json()['results'][0]
            print(f"  iNat name: {data['name']}")
            print(f"  iNat common: {data.get('preferred_common_name', '')}")
            print(f"  Rank: {data.get('rank', '')}")
            print(f"  Type: {data.get('iconic_taxon_name', '')}")
