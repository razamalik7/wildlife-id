import os
import json

DATA_DIR = './api/training_data_v2/train'  # Use relative path since we run from root
TAXONOMY_FILE = './api/taxonomy_hierarchy.json'

def audit():
    print("🐢 OOGWAY TAXONOMY AUDIT")
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory not found at {DATA_DIR}")
        # Try alternate path if moved
        if os.path.exists('./training_data_v2/train'):
            DATA_DIR_ALT = './training_data_v2/train'
            print(f"   Found it at {DATA_DIR_ALT}")
            folders = os.listdir(DATA_DIR_ALT)
        else:
            return
    else:
        folders = os.listdir(DATA_DIR)
        
    try:
        with open(TAXONOMY_FILE, 'r') as f:
            taxonomy = json.load(f)
    except Exception as e:
        print(f"❌ Error loading taxonomy: {e}")
        return

    # Invert mappings for easy lookup
    species_to_family = {}
    species_to_class = {}
    
    for family, species_list in taxonomy['family_to_species'].items():
        for s in species_list:
            species_to_family[s] = family
            
    for cls, species_list in taxonomy['class_to_species'].items():
        for s in species_list:
            species_to_class[s] = cls
            
    # Check each folder
    missing_family = []
    missing_class = []
    unknown_species = []
    
    print(f"\nScanning {len(folders)} species folders...")
    
    for folder in folders:
        # Ignore non-folders
        if not os.path.isdir(os.path.join(DATA_DIR if os.path.exists(DATA_DIR) else './training_data_v2/train', folder)):
            continue
            
        if folder not in species_to_family:
            missing_family.append(folder)
        if folder not in species_to_class:
            missing_class.append(folder)
            
    # Report
    if not missing_family and not missing_class:
        print("\n✅ PERFECT MATCH! All folder names exist in taxonomy.")
    else:
        print(f"\n⚠️ FOUND MISMATCHES (These will lack hierarchical loss benefits):")
        if missing_family:
            print(f"   Missing Family Map ({len(missing_family)}): {missing_family[:5]}...")
        if missing_class:
            print(f"   Missing Class Map ({len(missing_class)}): {missing_class[:5]}...")
        
        print("\nFix: Add these species keys to 'taxonomy_hierarchy.json' to ensure Oogway works correctly.")

if __name__ == "__main__":
    audit()
