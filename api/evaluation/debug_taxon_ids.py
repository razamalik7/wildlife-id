"""
Debug scraper - check what taxon IDs we're actually getting for problem species
"""
import requests
import json

problem_species = [
    'elk',
    'moose', 
    'arctic fox',
    'red fox',
    'gemsbok',
    'polar bear',  # Control - this one works
    'red-spotted newt',
    'spotted salamander'
]

def get_taxon_details(animal_name):
    """Get detailed taxon info from iNaturalist"""
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": animal_name, "rank": "species,subspecies", "per_page": 5}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            return results
    except Exception as e:
        print(f"Error: {e}")
    return []

print("="*70)
print("TAXON ID DEBUG - Checking what iNaturalist returns")
print("="*70)

for species in problem_species:
    print(f"\n{species.upper()}:")
    print("-"*70)
    
    results = get_taxon_details(species)
    
    if not results:
        print("  NO RESULTS FOUND")
        continue
    
    print(f"  Top result (what scraper uses):")
    top = results[0]
    print(f"    ID: {top['id']}")
    print(f"    Name: {top['name']}")
    print(f"    Common: {top.get('preferred_common_name', 'N/A')}")
    print(f"    Rank: {top.get('rank', 'N/A')}")
    print(f"    Matched Term: {top.get('matched_term', 'N/A')}")
    
    if len(results) > 1:
        print(f"\n  Other matches:")
        for i, result in enumerate(results[1:3], 2):
            print(f"    {i}. {result.get('name')} - {result.get('preferred_common_name')} (ID: {result['id']})")

print("\n" + "="*70)
print("FINDINGS")
print("="*70)
print("""
Check if:
1. Elk returns cervid family instead of species
2. Gemsbok returns plant genus instead of animal
3. Arctic fox gets confused with general fox queries
""")
