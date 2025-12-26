"""
Proper iNaturalist validation using taxon IDs from species_config.json
Uses iNat Vision API to identify images and compares to expected taxon_id
"""
import requests
import json
import os
from tqdm import tqdm
from collections import defaultdict
import time
import base64

DATA_DIR = 'training_data_cropped'
CONFIG_FILE = 'species_config.json'

# Load species config with taxon IDs
with open(CONFIG_FILE, 'r') as f:
    SPECIES_CONFIG = json.load(f)

# Build taxon_id mapping
SPECIES_TO_TAXON = {}
for entry in SPECIES_CONFIG:
    name = entry['name'].lower().replace(' ', '_')
    taxonomy = entry.get('taxonomy', {})
    taxon_id = taxonomy.get('taxon_id')
    
    if taxon_id:
        SPECIES_TO_TAXON[name] = taxon_id
    else:
        print(f"Warning: {name} missing taxon_id, skipping")

print(f"Loaded {len(SPECIES_TO_TAXON)} species with taxon IDs")

def get_inat_vision_prediction(image_path):
    """
    Use iNaturalist Vision API to identify an image
    Returns top prediction with taxon_id and score
    """
    url = "https://api.inaturalist.org/v1/computervision/score_image"
    
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Make API request
        response = requests.post(
            url,
            json={'image': img_data},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                top_result = results[0]
                return {
                    'taxon_id': top_result['taxon']['id'],
                    'name': top_result['taxon'].get('name', 'Unknown'),
                    'common_name': top_result['taxon'].get('preferred_common_name', ''),
                    'score': top_result.get('combined_score', 0)
                }
        
        return None
        
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return None

def validate_species_folder(species_name, max_images=20):
    """
    Validate a species folder using iNat Vision API
    Returns accuracy: % where predicted taxon_id matches expected
    """
    if species_name not in SPECIES_TO_TAXON:
        print(f"Warning: {species_name} not in config, skipping")
        return None
    
    expected_taxon_id = SPECIES_TO_TAXON[species_name]
    
    results = {
        'correct': 0,
        'total': 0,
        'predictions': defaultdict(int),
        'expected_taxon_id': expected_taxon_id
    }
    
    # Check train folder (sample for speed)
    species_dir = os.path.join(DATA_DIR, 'train', species_name)
    
    if not os.path.exists(species_dir):
        return None
    
    images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sample for speed
    if len(images) > max_images:
        import random
        images = random.sample(images, max_images)
    
    for img_name in images:
        img_path = os.path.join(species_dir, img_name)
        
        # Get iNat prediction
        pred = get_inat_vision_prediction(img_path)
        
        if pred is None:
            continue
        
        results['total'] += 1
        predicted_taxon_id = pred['taxon_id']
        results['predictions'][predicted_taxon_id] += 1
        
        # Check if correct
        if predicted_taxon_id == expected_taxon_id:
            results['correct'] += 1
        
        # Rate limit
        time.sleep(0.5)  # Be nice to iNat API
    
    return results

if __name__ == '__main__':
    print("="*70)
    print("INAT VISION API DATASET VALIDATION")
    print("="*70)
    print("Using species taxon IDs from species_config.json")
    print("WARNING: This will be slow due to API rate limits (~20 images/min)")
    print("="*70)
    
    # Get all species
    train_dir = os.path.join(DATA_DIR, 'train')
    all_species = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"\nFound {len(all_species)} species folders")
    print("Sampling 20 images per species (will take ~100 minutes for all)")
    
    response = input("\nProceed with full validation? (yes/no/test): ").strip().lower()
    
    if response == 'test':
        # Test only problem species
        test_species = ['elk', 'arctic_fox', 'gemsbok', 'jaguar']
        all_species = [s for s in all_species if s in test_species]
        print(f"\nTesting only: {test_species}")
    elif response != 'yes':
        print("Cancelled")
        exit()
    
    # Validate all species
    species_results = {}
    
    for species in tqdm(all_species, desc="Validating"):
        results = validate_species_folder(species, max_images=20)
        if results and results['total'] > 0:
            accuracy = results['correct'] / results['total']
            species_results[species] = {
                'accuracy': accuracy,
                'correct': results['correct'],
                'total': results['total'],
                'expected_taxon_id': results['expected_taxon_id'],
                'top_predicted_taxon': max(results['predictions'].items(), key=lambda x: x[1])[0] if results['predictions'] else None
            }
    
    # Report
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    # Sort by accuracy
    sorted_results = sorted(species_results.items(), key=lambda x: x[1]['accuracy'])
    
    print("\nWORST 20 SPECIES:")
    print(f"{'Species':<35} {'Accuracy':>10} {'Correct/Total':>15} {'Issue':>20}")
    print("-"*70)
    
    for species, data in sorted_results[:20]:
        acc = data['accuracy']
        correct = data['correct']
        total = data['total']
        
        flag = "🔴" if acc < 0.5 else "🟡" if acc < 0.8 else ""
        issue = "CONTAMINATION" if acc < 0.5 else ""
        
        print(f"{flag} {species:<35} {acc:>9.1%} {correct:>7}/{total:<7} {issue:>20}")
    
    print("\n\nBEST 10 SPECIES:")
    print(f"{'Species':<35} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-"*70)
    
    for species, data in sorted_results[-10:]:
        acc = data['accuracy']
        correct = data['correct']
        total = data['total']
        print(f"✓ {species:<35} {acc:>9.1%} {correct:>7}/{total:<7}")
    
    # Save report
    with open('inat_vision_validation_report.json', 'w') as f:
        json.dump(species_results, f, indent=2)
    
    print(f"\n📁 Full report saved to: inat_vision_validation_report.json")
    
    # Summary stats
    contaminated = [s for s, d in species_results.items() if d['accuracy'] < 0.5]
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total species validated: {len(species_results)}")
    print(f"Heavily contaminated (<50% acc): {len(contaminated)}")
    
    if contaminated:
        print(f"\nContaminated species:")
        for species in contaminated:
            data = species_results[species]
            print(f"  - {species}: {data['accuracy']:.1%} accuracy")
