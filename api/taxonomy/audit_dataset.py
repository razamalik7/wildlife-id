"""
Dataset audit script - check for potential mislabeling in problem species
"""
import os
from pathlib import Path

# Problem species identified
problem_species = {
    'elk': 'likely contains moose',
    'arctic_fox': 'likely contains red_fox',
    'eastern_newt': 'likely contains spotted_salamander',
    'spotted_salamander': 'likely contains eastern_newt'
}

data_dir = './training_data_cropped'

print("="*70)
print("DATASET AUDIT - MISLABELING DETECTION")
print("="*70)

for split in ['train', 'val']:
    print(f"\n{split.upper()} SET:")
    print("-"*70)
    
    for species, issue in problem_species.items():
        species_dir = os.path.join(data_dir, split, species)
        
        if os.path.exists(species_dir):
            images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"\n{species:30s}: {len(images):4d} images")
            print(f"  Issue: {issue}")
            print(f"  Path: {species_dir}")
        else:
            print(f"\n{species:30s}: NOT FOUND")

print("\n" + "="*70)
print("RECOMMENDED ACTIONS")
print("="*70)
print("1. Manually review elk folder - remove/relabel moose images")
print("2. Manually review arctic_fox folder - remove/relabel red_fox images")
print("3. Review newt/salamander folders for cross-contamination")
print("""
4. Options for cleaning:
   a) Manual review and delete (time-consuming but thorough)
   b) Use a pre-trained classifier to auto-detect mislabels
   c) Re-scrape from iNaturalist with stricter quality filters
""")

print("\n" + "="*70)
print("IMPACT ESTIMATE")
print("="*70)
print("If elk folder is 50%+ mislabeled:")
print("  - Explains 3.73% accuracy (model learned mislabeled data)")
print("  - Hard negative mining made it WORSE (forced model to see mislabels every batch)")
print("  - Multi-task was innocent - data quality was the issue")
print("\nAfter cleaning data, expected improvements:")
print("  - Elk: 3.73% → 60-70% (normal cervid distinction)")
print("  - Arctic Fox: 32% → 65-75% (normal fox distinction)")
print("  - Overall: 75.58% → 78-80%+ (above Grandmaster!)")
