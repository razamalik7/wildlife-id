"""
Analyze scraping status of training_data_v2 to identify incomplete species.
"""
import os
import json
from pathlib import Path

OUTPUT_DIR = 'training_data_v2'
TRAIN_TARGET = 600
VAL_TARGET = 70

# Load species config
with open('species_config.json', 'r') as f:
    config = json.load(f)

species_map = {}
for entry in config:
    name = entry['name']
    folder_name = name.lower().replace(' ', '_').replace('-', '-')
    species_map[folder_name] = name

print("=" * 80)
print("SCRAPING STATUS ANALYSIS - training_data_v2")
print("=" * 80)
print(f"Target: {TRAIN_TARGET} train + {VAL_TARGET} val = {TRAIN_TARGET + VAL_TARGET} per species")
print("=" * 80)
print()

complete = []
incomplete = []

for folder_name, species_name in sorted(species_map.items()):
    train_dir = Path(OUTPUT_DIR) / 'train' / folder_name
    val_dir = Path(OUTPUT_DIR) / 'val' / folder_name
    
    train_count = 0
    val_count = 0
    
    if train_dir.exists():
        train_count = len([f for f in train_dir.glob('*.jpg')])
    
    if val_dir.exists():
        val_count = len([f for f in val_dir.glob('*.jpg')])
    
    total = train_count + val_count
    status = "✓" if (train_count >= TRAIN_TARGET and val_count >= VAL_TARGET) else "⚠️"
    
    if train_count >= TRAIN_TARGET and val_count >= VAL_TARGET:
        complete.append(folder_name)
    else:
        incomplete.append((folder_name, species_name, train_count, val_count, total))
        print(f"{status} {species_name:40} | Train: {train_count:3}/{TRAIN_TARGET} | Val: {val_count:2}/{VAL_TARGET} | Total: {total:3}/{TRAIN_TARGET + VAL_TARGET}")

print()
print("=" * 80)
print(f"SUMMARY: {len(complete)} complete, {len(incomplete)} incomplete")
print("=" * 80)
print()

if incomplete:
    print("INCOMPLETE SPECIES (sorted by deficit):")
    print("-" * 80)
    incomplete_sorted = sorted(incomplete, key=lambda x: x[4])  # Sort by total count
    for folder, name, train, val, total in incomplete_sorted:
        deficit = (TRAIN_TARGET + VAL_TARGET) - total
        print(f"  {name:40} | Deficit: {deficit:3} | Train: {train:3} | Val: {val:2}")
