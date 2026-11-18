import requests
import json

# Check moose taxon_id
with open('species_config.json') as f:
    config = json.load(f)

moose = [e for e in config if 'moose' in e['name'].lower()][0]
taxon_id = moose['taxonomy']['taxon_id']

print(f"Moose taxon_id: {taxon_id}")

# Test API call
r = requests.get(
    'https://api.inaturalist.org/v1/observations',
    params={
        'taxon_id': taxon_id,
        'quality_grade': 'research',
        'photos': 'true',
        'per_page': 5
    }
)

print(f"\nAPI Status: {r.status_code}")
results = r.json().get('results', [])
print(f"Got {len(results)} observations")

if results:
    obs = results[0]
    print(f"\nFirst observation:")
    print(f"  ID: {obs['id']}")
    print(f"  Photos: {len(obs.get('photos', []))}")
    if obs.get('photos'):
        print(f"  Photo URL: {obs['photos'][0]['url']}")
