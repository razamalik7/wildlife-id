"""
Rename species in species_config.json:
- Eastern Box Turtle -> Common Box Turtle
- Dall Sheep -> Thinhorn Sheep
- Red-spotted Newt -> Eastern Newt
"""
import json

# Load config
config = json.load(open('species_config.json'))

renames = {
    'Eastern Box Turtle': 'Common Box Turtle',
    'Dall Sheep': 'Thinhorn Sheep',
    'Red-spotted Newt': 'Eastern Newt'
}

renamed_count = 0

for entry in config:
    old_name = entry['name']
    if old_name in renames:
        new_name = renames[old_name]
        entry['name'] = new_name
        
        # Also update common_name in taxonomy if present
        if 'taxonomy' in entry and 'common_name' in entry['taxonomy']:
            entry['taxonomy']['common_name'] = new_name
        
        print(f"  Renamed: {old_name} -> {new_name}")
        renamed_count += 1

# Save
with open('species_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print(f"\n✓ Renamed {renamed_count} species in species_config.json")
