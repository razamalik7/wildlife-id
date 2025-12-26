import requests
import json

# Search for Wolverine specifically
print("Searching for Wolverine (Gulo gulo) with mammal filter...")

r = requests.get(
    'https://api.inaturalist.org/v1/taxa',
    params={
        'q': 'Wolverine',
        'iconic_taxa': 'Mammalia',
        'rank': 'species',
        'per_page': 10
    },
    timeout=15
)

if r.status_code == 200:
    results = r.json().get('results', [])
    print(f"Found {len(results)} mammal results:")
    
    for result in results:
        print(f"  - {result['name']} ({result.get('preferred_common_name', '')}) ID: {result['id']}")
        
    # Also try direct Gulo genus search
    print("\nSearching Gulo genus...")
    r2 = requests.get(
        'https://api.inaturalist.org/v1/taxa',
        params={
            'q': 'Gulo',
            'per_page': 10
        },
        timeout=15
    )
    
    if r2.status_code == 200:
        results2 = r2.json().get('results', [])
        for result in results2:
            iconic = result.get('iconic_taxon_name', '')
            if iconic == 'Mammalia':
                print(f"  ✓ {result['name']} ({result.get('preferred_common_name', '')}) ID: {result['id']} Type: {iconic}")
