"""
Comprehensive debug of hard negative mining and compare with Hero/Grandmaster
"""
import torch
import numpy as np
import json
from collections import defaultdict

print("="*70)
print("COMPARING ELK PERFORMANCE ACROSS MODELS")
print("="*70)

# Load Oogway diagnostic
oogway_data = json.load(open('oogway_diagnostic_data.json'))

elk_oogway = None
arctic_fox_oogway = None
for species in oogway_data['per_class_accuracy']:
    if species['species'] == 'elk':
        elk_oogway = species
    if species['species'] == 'arctic_fox':
        arctic_fox_oogway = species

print(f"\nOogway B3:")
print(f"  Elk: {elk_oogway['accuracy']:.2%} ({elk_oogway['correct']}/{elk_oogway['total']})")
print(f"  Arctic Fox: {arctic_fox_oogway['accuracy']:.2%} ({arctic_fox_oogway['correct']}/{arctic_fox_oogway['total']})")

# Try to load Hero model for comparison
try:
    hero_ckpt = torch.load('b3_hero_best.pth', map_location='cpu')
    print(f"\nHero B3 found:")
    print(f"  Overall: {hero_ckpt.get('accuracy', 'N/A')}")
except:
    print("\nHero B3 not found")

# Analyze Oogway confusion matrix for elk
print("\n" + "="*70)
print("ELK CONFUSION BREAKDOWN (Oogway)")
print("="*70)

confusion_matrix = np.array(oogway_data['confusion_matrix'])
classes = [s['species'] for s in sorted(oogway_data['per_class_accuracy'], key=lambda x: oogway_data['per_class_accuracy'].index(x))]

# This won't work perfectly since classes aren't in original order, but let's try
# Actually, let's reload properly
from train_oogway import WildlifeGeoDataset
from torchvision import transforms

val_transform = transforms.Compose([
    transforms.Resize(330),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_ds = WildlifeGeoDataset('./training_data_cropped/val', val_transform, is_train=False)
elk_idx = val_ds.classes.index('elk')
moose_idx = val_ds.classes.index('moose')

print(f"\nElk true samples: {confusion_matrix[elk_idx].sum()}")
print(f"Elk correctly classified: {confusion_matrix[elk_idx, elk_idx]}")
print(f"Elk misclassified as moose: {confusion_matrix[elk_idx, moose_idx]}")
print(f"Elk misclassified as other: {confusion_matrix[elk_idx].sum() - confusion_matrix[elk_idx, elk_idx] - confusion_matrix[elk_idx, moose_idx]}")

print(f"\nMoose true samples: {confusion_matrix[moose_idx].sum()}")
print(f"Moose correctly classified: {confusion_matrix[moose_idx, moose_idx]}")
print(f"Moose misclassified as elk: {confusion_matrix[moose_idx, elk_idx]}")

# Check hard negative mining sampler
print("\n" + "="*70)
print("HARD NEGATIVE MINING SAMPLER DEBUG")
print("="*70)

from train_oogway import HardNegativeSampler, CONFUSED_PAIRS, PAIR_PROBS
from torch.utils.data import DataLoader

train_ds = WildlifeGeoDataset('./training_data_cropped/train', val_transform, is_train=True)
sampler = HardNegativeSampler(train_ds, batch_size=16, num_pairs_per_batch=3)

print(f"\nValid confused pairs: {len(sampler.valid_pairs)}")
print(f"Elk in class_indices: {'elk' in sampler.class_indices}")
print(f"Moose in class_indices: {'moose' in sampler.class_indices}")

if 'elk' in sampler.class_indices:
    print(f"Elk training samples: {len(sampler.class_indices['elk'])}")
if 'moose' in sampler.class_indices:
    print(f"Moose training samples: {len(sampler.class_indices['moose'])}")

# Check if elk-moose pair is in valid pairs
elk_moose_in_valid = ('elk', 'moose') in sampler.valid_pairs or ('moose', 'elk') in sampler.valid_pairs
print(f"Elk-moose pair in valid pairs: {elk_moose_in_valid}")

# Sample 100 batches and count how many have elk/moose
elk_batch_count = 0
moose_batch_count = 0
elk_moose_together = 0

for batch_num, batch_indices in enumerate(sampler):
    if batch_num >= 100:
        break
    
    has_elk = False
    has_moose = False
    
    for idx in batch_indices:
        _, _, _, class_name = train_ds.samples[idx]
        if class_name == 'elk':
            has_elk = True
        if class_name == 'moose':
            has_moose = True
    
    if has_elk:
        elk_batch_count += 1
    if has_moose:
        moose_batch_count += 1
    if has_elk and has_moose:
        elk_moose_together += 1

print(f"\nIn first 100 batches:")
print(f"  Batches with elk: {elk_batch_count}")
print(f"  Batches with moose: {moose_batch_count}")
print(f"  Batches with both: {elk_moose_together}")
print(f"  Expected (with 3 pairs/batch): ~100 batches should have both")

if elk_moose_together < 50:
    print("\n⚠️  WARNING: Elk-moose pairs appearing less than expected!")
    print("   Hard negative mining may not be working correctly")

print("\n" + "="*70)
print("HYPOTHESIS")
print("="*70)
print("If elk/moose appear together in every batch but elk accuracy is 3.73%:")
print("1. Model learned to ALWAYS predict moose when seeing cervids")
print("2. Multi-task + hierarchical loss penalized cross-family more than within-family")
print("3. Hard negative mining taught 'similarity' not 'distinction'")
