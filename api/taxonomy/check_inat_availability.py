"""
Check iNaturalist data availability for incomplete species.
"""
import requests
import time

INAT_API = 'https://api.inaturalist.org/v1'

# Species with issues
incomplete_species = [
    ("Black-footed Ferret", 41949),
    ("Veiled Chameleon", 36223),
    ("Wolverine", 41869),
    ("Eastern Hellbender", 66693),
    ("Walrus", 41530),
    ("Burmese Python", 238252),
    ("Gila Monster", 26293),
    ("Grizzly Bear", 125461),
]

print("=" * 80)
print("iNATURALIST DATA AVAILABILITY CHECK")
print("=" * 80)
print()

for name, taxon_id in incomplete_species:
    try:
        r = requests.get(
            f'{INAT_API}/observations',
            params={
                'taxon_id': taxon_id,
                'quality_grade': 'research',
                'photos': 'true',
                'per_page': 1
            },
            timeout=10
        )
        
        if r.status_code == 200:
            data = r.json()
            total_results = data.get('total_results', 0)
            print(f"{name:40} | Taxon: {taxon_id:7} | Observations: {total_results:>6,}")
        else:
            print(f"{name:40} | ERROR: HTTP {r.status_code}")
        
        time.sleep(0.5)  # Rate limiting
        
    except Exception as e:
        print(f"{name:40} | ERROR: {e}")

print()
print("=" * 80)
print("NOTES:")
print("- Species with <1000 observations might struggle to reach 670 quality images")
print("- Rare/endangered species (Black-footed Ferret, Wolverine) have limited data")
print("- Exotic species (Veiled Chameleon) may have mostly captive observations")
print("=" * 80)
