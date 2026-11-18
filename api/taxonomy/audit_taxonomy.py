"""
Full taxonomy audit of all taxon_ids
Shows: Name | taxon_id | Class | Order | Family | Genus | Species
"""
import requests
import json
import time
from tqdm import tqdm

with open('species_config.json', 'r') as f:
    species_config = json.load(f)

print("="*120)
print("FULL TAXONOMY AUDIT")
print("="*120)

results = []

for entry in tqdm(species_config, desc="Fetching"):
    name = entry['name']
    taxonomy = entry.get('taxonomy', {})
    taxon_id = taxonomy.get('taxon_id')
    
    row = {
        'our_name': name,
        'taxon_id': taxon_id,
        'class': '',
        'order': '',
        'family': '',
        'genus': '',
        'species': '',
        'inat_common': '',
        'rank': '',
        'iconic': ''
    }
    
    if not taxon_id:
        row['species'] = 'MISSING ID'
        results.append(row)
        continue
    
    try:
        r = requests.get(f'https://api.inaturalist.org/v1/taxa/{taxon_id}', timeout=10)
        
        if r.status_code == 200:
            data = r.json().get('results', [{}])[0]
            
            row['inat_common'] = data.get('preferred_common_name', '')
            row['rank'] = data.get('rank', '')
            row['iconic'] = data.get('iconic_taxon_name', '')
            row['species'] = data.get('name', '')
            
            # Get full ancestry
            ancestors = data.get('ancestors', [])
            for anc in ancestors:
                anc_rank = anc.get('rank', '')
                anc_name = anc.get('name', '')
                if anc_rank == 'class':
                    row['class'] = anc_name
                elif anc_rank == 'order':
                    row['order'] = anc_name
                elif anc_rank == 'family':
                    row['family'] = anc_name
                elif anc_rank == 'genus':
                    row['genus'] = anc_name
        else:
            row['species'] = f'HTTP {r.status_code}'
            
    except Exception as e:
        row['species'] = str(e)[:30]
    
    results.append(row)
    time.sleep(0.25)

# Print table
print("\n")
print(f"{'Our Name':<25} {'ID':>8} {'Rank':<10} {'Class':<15} {'Order':<20} {'Family':<20} {'Genus':<15} {'iNat Species':<25} {'Iconic':<10}")
print("-"*160)

for r in results:
    flag = ''
    # Flag issues
    if r['rank'] not in ['species', 'subspecies', '']:
        flag = '⚠️GENUS'
    if r['iconic'] == 'Plantae':
        flag = '🌱PLANT'
    if r['species'] == 'MISSING ID':
        flag = '❓MISSING'
    
    print(f"{r['our_name']:<25} {str(r['taxon_id']):>8} {r['rank']:<10} {r['class']:<15} {r['order']:<20} {r['family']:<20} {r['genus']:<15} {r['species']:<25} {r['iconic']:<10} {flag}")

# Save to file for easier review
with open('taxonomy_audit.json', 'w') as f:
    json.dump(results, f, indent=2)

# Also save as CSV for spreadsheet view
with open('taxonomy_audit.csv', 'w') as f:
    f.write("Our Name,taxon_id,Rank,Class,Order,Family,Genus,iNat Species,Iconic,iNat Common\n")
    for r in results:
        f.write(f"{r['our_name']},{r['taxon_id']},{r['rank']},{r['class']},{r['order']},{r['family']},{r['genus']},{r['species']},{r['iconic']},{r['inat_common']}\n")

print(f"\n📁 Saved to: taxonomy_audit.json and taxonomy_audit.csv")
