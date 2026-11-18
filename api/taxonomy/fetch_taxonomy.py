"""
Taxonomy Enrichment Script
==========================
Fetches complete taxonomic hierarchy from iNaturalist API and 
updates species_config.json with detailed taxonomy.

Hierarchy levels:
- Kingdom (Animalia)
- Phylum (Chordata)
- Class (Mammalia, Aves, Reptilia, Amphibia)
- Order (Carnivora, Artiodactyla, Rodentia, etc.)
- Family (Canidae, Felidae, Ursidae, Cervidae, etc.)
- Genus
- Species

This enables hierarchical classification where the model can:
1. First classify into Class (Mammal/Bird/Reptile/Amphibian)
2. Then into Family (Canid/Felid/Bear/Deer/etc.)
3. Finally into exact species

Usage:
  python fetch_taxonomy.py
"""

import json
import requests
import time
from tqdm import tqdm

INAT_API_BASE = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "WildlifeIDApp/1.0"}

def get_taxon_info(scientific_name):
    """
    Fetch complete taxonomic hierarchy from iNaturalist.
    Requires two API calls: search for taxon ID, then fetch full details.
    """
    try:
        # Step 1: Search for the taxon to get its ID
        url = f"{INAT_API_BASE}/taxa"
        params = {"q": scientific_name, "per_page": 1}
        
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        
        results = resp.json().get('results', [])
        if not results:
            return None
        
        taxon_id = results[0].get('id')
        if not taxon_id:
            return None
        
        # Step 2: Fetch full taxon details including ancestry
        detail_url = f"{INAT_API_BASE}/taxa/{taxon_id}"
        resp2 = requests.get(detail_url, headers=HEADERS, timeout=10)
        if resp2.status_code != 200:
            return None
        
        detail_results = resp2.json().get('results', [])
        if not detail_results:
            return None
        
        taxon = detail_results[0]
        
        # Build taxonomy from ancestors
        taxonomy = {
            'kingdom': None,
            'phylum': None,
            'class': None,
            'order': None,
            'family': None,
            'genus': None,
            'species': taxon.get('name'),
            'common_name': taxon.get('preferred_common_name'),
            'taxon_id': taxon.get('id')
        }
        
        # Parse ancestors (from highest to lowest rank)
        ancestors = taxon.get('ancestors', [])
        for ancestor in ancestors:
            rank = ancestor.get('rank', '').lower()
            name = ancestor.get('name')
            
            if rank == 'kingdom':
                taxonomy['kingdom'] = name
            elif rank == 'phylum':
                taxonomy['phylum'] = name
            elif rank == 'class':
                taxonomy['class'] = name
            elif rank == 'order':
                taxonomy['order'] = name
            elif rank == 'family':
                taxonomy['family'] = name
            elif rank == 'genus':
                taxonomy['genus'] = name
        
        return taxonomy
    
    except Exception as e:
        print(f"\n      Error fetching {scientific_name}: {e}")
        return None


def enrich_species_config():
    """
    Load species_config.json, fetch taxonomy for each species,
    and save updated version.
    """
    print("🦁 TAXONOMY ENRICHMENT SCRIPT")
    print("   Fetching complete taxonomic hierarchy from iNaturalist")
    
    # Load current config
    with open('species_config.json', 'r') as f:
        species_list = json.load(f)
    
    print(f"📂 Loaded {len(species_list)} species from species_config.json")
    print("\n🌐 Fetching taxonomy data from iNaturalist API...\n")
    
    updated_count = 0
    failed_count = 0
    
    for species in tqdm(species_list, desc="Fetching taxonomy"):
        scientific_name = species.get('scientific_name')
        
        if not scientific_name:
            failed_count += 1
            continue
        
        # Check if already has VALID taxonomy (not from failed previous run)
        existing_tax = species.get('taxonomy', {})
        if existing_tax.get('family') is not None and existing_tax.get('class') is not None:
            continue  # Already enriched successfully
        
        taxonomy = get_taxon_info(scientific_name)
        
        if taxonomy and taxonomy.get('family'):
            species['taxonomy'] = taxonomy
            updated_count += 1
        else:
            # Set defaults if lookup failed
            species['taxonomy'] = {
                'kingdom': 'Animalia',
                'phylum': 'Chordata',
                'class': None,
                'order': None,
                'family': None,
                'genus': scientific_name.split()[0] if ' ' in scientific_name else None,
                'species': scientific_name
            }
            failed_count += 1
        
        # Rate limiting (0.5s between each species - 2 API calls each)
        time.sleep(0.5)
    
    # Save updated config
    with open('species_config.json', 'w') as f:
        json.dump(species_list, f, indent=4)
    
    print(f"\n✅ Updated {updated_count} species with taxonomy")
    if failed_count > 0:
        print(f"⚠️ Failed to fetch {failed_count} species (using defaults)")
    
    # Print summary of taxonomy distribution
    print("\n📊 TAXONOMY DISTRIBUTION:")
    
    # Count by class
    class_counts = {}
    for species in species_list:
        cls = species.get('taxonomy', {}).get('class', 'Unknown')
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\n   By Class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"      {cls}: {count}")
    
    # Count by family
    family_counts = {}
    for species in species_list:
        family = species.get('taxonomy', {}).get('family', 'Unknown')
        family_counts[family] = family_counts.get(family, 0) + 1
    
    print("\n   By Family (top 15):")
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"      {family}: {count}")
    
    # Count by order
    order_counts = {}
    for species in species_list:
        order = species.get('taxonomy', {}).get('order', 'Unknown')
        order_counts[order] = order_counts.get(order, 0) + 1
    
    print("\n   By Order (top 10):")
    for order, count in sorted(order_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"      {order}: {count}")
    
    print(f"\n💾 Saved enriched config to species_config.json")
    
    # Generate taxonomy summary for hierarchy script
    print("\n📋 GENERATING HIERARCHY MAPPING...")
    
    class_to_species = {}
    family_to_species = {}
    
    for species in species_list:
        name = species.get('name', '').lower().replace(' ', '_')
        tax = species.get('taxonomy', {})
        
        cls = tax.get('class', 'Unknown')
        family = tax.get('family', 'Unknown')
        
        if cls not in class_to_species:
            class_to_species[cls] = []
        class_to_species[cls].append(name)
        
        if family not in family_to_species:
            family_to_species[family] = []
        family_to_species[family].append(name)
    
    # Save mapping file for hierarchy training
    hierarchy_map = {
        'class_to_species': class_to_species,
        'family_to_species': family_to_species,
        'class_names': list(class_to_species.keys()),
        'family_names': list(family_to_species.keys())
    }
    
    with open('taxonomy_hierarchy.json', 'w') as f:
        json.dump(hierarchy_map, f, indent=2)
    
    print(f"💾 Saved taxonomy_hierarchy.json for hierarchy training")
    print(f"\n   Classes: {len(class_to_species)}")
    print(f"   Families: {len(family_to_species)}")


if __name__ == "__main__":
    enrich_species_config()
