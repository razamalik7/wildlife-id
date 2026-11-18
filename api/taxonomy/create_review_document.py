"""
Create comprehensive side-by-side comparison document
for manual review of all 100 species
Also fixes Wolverine taxon_id
"""
import requests
import json
import time
from tqdm import tqdm

# First fix Wolverine
print("Fixing Wolverine taxon_id...")
config = json.load(open('species_config.json'))

for entry in config:
    if entry['name'] == 'Wolverine':
        old_id = entry['taxonomy'].get('taxon_id')
        entry['taxonomy']['taxon_id'] = 41852
        print(f"  Updated Wolverine: {old_id} -> 41852")
        break

with open('species_config.json', 'w') as f:
    json.dump(config, f, indent=4)

# Now create comprehensive review document
print("\nCreating comprehensive review document...")

# Reload updated config
config = json.load(open('species_config.json'))

results = []

for entry in tqdm(config, desc="Fetching from iNat"):
    name = entry['name']
    our_scientific = entry.get('scientific_name', entry.get('taxonomy', {}).get('species', ''))
    taxon_id = entry.get('taxonomy', {}).get('taxon_id')
    
    row = {
        'our_name': name,
        'our_scientific': our_scientific,
        'taxon_id': taxon_id,
        'inat_scientific': '',
        'inat_common': '',
        'inat_rank': '',
        'inat_type': '',
        'match': '❓',
        'notes': ''
    }
    
    if not taxon_id:
        row['match'] = '❌'
        row['notes'] = 'MISSING TAXON_ID'
        results.append(row)
        continue
    
    try:
        r = requests.get(f'https://api.inaturalist.org/v1/taxa/{taxon_id}', timeout=15)
        
        if r.status_code == 200:
            data = r.json().get('results', [{}])[0]
            
            row['inat_scientific'] = data.get('name', '')
            row['inat_common'] = data.get('preferred_common_name', '')
            row['inat_rank'] = data.get('rank', '')
            row['inat_type'] = data.get('iconic_taxon_name', '')
            
            # Check for match
            name_lower = name.lower().replace('-', ' ').replace('_', ' ')
            inat_common_lower = row['inat_common'].lower().replace('-', ' ').replace('_', ' ')
            
            if name_lower == inat_common_lower or name_lower in inat_common_lower or inat_common_lower in name_lower:
                row['match'] = '✓'
            elif row['inat_rank'] not in ['species', 'subspecies']:
                row['match'] = '⚠️'
                row['notes'] = f"WRONG RANK: {row['inat_rank']}"
            elif row['inat_type'] not in ['Mammalia', 'Aves', 'Reptilia', 'Amphibia']:
                row['match'] = '❌'
                row['notes'] = f"NOT ANIMAL: {row['inat_type']}"
            else:
                row['match'] = '🔍'
                row['notes'] = 'Name mismatch - REVIEW'
                
        elif r.status_code == 429:
            row['notes'] = 'Rate limited - retry'
        else:
            row['notes'] = f'HTTP {r.status_code}'
            
    except Exception as e:
        row['notes'] = str(e)[:30]
    
    results.append(row)
    time.sleep(0.5)

# Create markdown document
md_content = """# Taxon ID Verification Report

## Summary
- Total Species: 100
- ✓ Match: Names match correctly
- ⚠️ Alert: Wrong rank (genus instead of species)
- 🔍 Review: Names don't match but might be subspecies/alternate name
- ❌ Error: Not an animal or other critical issue

---

## Full Species List

| # | Our Name | Our Scientific | taxon_id | iNat Scientific | iNat Common | Rank | Type | Status | Notes |
|---|----------|----------------|----------|-----------------|-------------|------|------|--------|-------|
"""

for i, r in enumerate(results, 1):
    md_content += f"| {i} | {r['our_name']} | {r['our_scientific']} | {r['taxon_id']} | {r['inat_scientific']} | {r['inat_common']} | {r['inat_rank']} | {r['inat_type']} | {r['match']} | {r['notes']} |\n"

# Add summary section
matches = sum(1 for r in results if r['match'] == '✓')
alerts = sum(1 for r in results if r['match'] == '⚠️')
reviews = sum(1 for r in results if r['match'] == '🔍')
errors = sum(1 for r in results if r['match'] == '❌')

md_content += f"""

---

## Summary Statistics
- ✓ Matches: {matches}
- ⚠️ Alerts: {alerts}
- 🔍 Needs Review: {reviews}
- ❌ Errors: {errors}

---

## Items Needing Attention

### ⚠️ Wrong Rank (Genus instead of Species)
"""

for r in results:
    if r['match'] == '⚠️':
        md_content += f"- **{r['our_name']}** (ID: {r['taxon_id']}): Returns `{r['inat_scientific']}` which is a {r['inat_rank']}\n"

md_content += """

### 🔍 Name Mismatch (Need Review)
"""

for r in results:
    if r['match'] == '🔍':
        md_content += f"- **{r['our_name']}** (ID: {r['taxon_id']}): Returns `{r['inat_common']}` ({r['inat_scientific']})\n"

md_content += """

### ❌ Critical Errors
"""

for r in results:
    if r['match'] == '❌':
        md_content += f"- **{r['our_name']}** (ID: {r['taxon_id']}): {r['notes']}\n"

# Save document
with open('taxon_review_document.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

# Also save as JSON for programmatic access
with open('taxon_review_data.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Documents saved:")
print(f"  - taxon_review_document.md (for manual review)")
print(f"  - taxon_review_data.json (raw data)")
print(f"\nSummary: {matches} matches, {alerts} alerts, {reviews} need review, {errors} errors")
