"""
Verify all taxon_ids in species_config.json against iNaturalist API
Checks:
1. iconic_taxon_name = Mammalia, Aves, Reptilia, Amphibia (NOT Plantae!)
2. rank = species or subspecies
"""
import requests
import json
import time
from tqdm import tqdm

with open('species_config.json', 'r') as f:
    species_config = json.load(f)

print("="*70)
print("TAXON ID VERIFICATION")
print("="*70)
print(f"Total species: {len(species_config)}")
print("Checking for: Animals only (not plants), Species rank")
print("="*70)

issues = []
missing = []
verified = []

VALID_TAXONS = ['Mammalia', 'Aves', 'Reptilia', 'Amphibia']

for entry in tqdm(species_config, desc="Verifying"):
    name = entry['name']
    taxonomy = entry.get('taxonomy', {})
    taxon_id = taxonomy.get('taxon_id')
    
    if not taxon_id:
        missing.append(name)
        continue
    
    try:
        r = requests.get(f'https://api.inaturalist.org/v1/taxa/{taxon_id}', timeout=10)
        
        if r.status_code == 200:
            result = r.json().get('results', [])
            if result:
                data = result[0]
                inat_name = data.get('name', '')
                inat_common = data.get('preferred_common_name', '')
                iconic_taxon = data.get('iconic_taxon_name', '')
                rank = data.get('rank', '')
                
                # Validation checks
                is_animal = iconic_taxon in VALID_TAXONS
                is_species = rank in ['species', 'subspecies']
                
                if is_animal and is_species:
                    verified.append({
                        'name': name,
                        'taxon_id': taxon_id,
                        'inat_name': inat_name,
                        'iconic_taxon': iconic_taxon
                    })
                else:
                    issues.append({
                        'name': name,
                        'taxon_id': taxon_id,
                        'inat_name': inat_name,
                        'inat_common': inat_common,
                        'iconic_taxon': iconic_taxon,
                        'rank': rank,
                        'problem': 'PLANT!' if not is_animal else f'Bad rank: {rank}'
                    })
            else:
                issues.append({'name': name, 'taxon_id': taxon_id, 'problem': 'No results'})
        else:
            issues.append({'name': name, 'taxon_id': taxon_id, 'problem': f'HTTP {r.status_code}'})
            
    except Exception as e:
        issues.append({'name': name, 'taxon_id': taxon_id, 'problem': str(e)})
    
    time.sleep(0.2)

# Report
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"✓ Verified (animals, species rank): {len(verified)}")
print(f"⚠ Missing taxon_id: {len(missing)}")
print(f"❌ Issues: {len(issues)}")

if missing:
    print(f"\nMISSING TAXON_ID:")
    for name in missing:
        print(f"  - {name}")

if issues:
    print(f"\n🚨 ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue['name']} (ID: {issue['taxon_id']})")
        if 'iconic_taxon' in issue:
            print(f"      iNat: {issue['inat_name']} | Type: {issue['iconic_taxon']} | Rank: {issue['rank']}")
        print(f"      Problem: {issue['problem']}")

# Save report
with open('taxon_verification_report.json', 'w') as f:
    json.dump({'verified': verified, 'missing': missing, 'issues': issues}, f, indent=2)

print(f"\n📁 Report saved to: taxon_verification_report.json")
