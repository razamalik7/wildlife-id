"""
Check if hard negative sampler is actually working correctly
"""
import torch
from train_oogway import WildlifeGeoDataset, HardNegativeSampler, CONFUSED_PAIRS
from torchvision import transforms

val_transform = transforms.Compose([
    transforms.Resize(330),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = WildlifeGeoDataset('./training_data_cropped/train', val_transform, is_train=True)

print(f"Dataset has {len(train_ds)} samples")
print(f"Dataset classes: {len(train_ds.classes)}")
print(f"Sample structure: {train_ds.samples[0]}")

# Check if hard negative sampler builds class indices correctly
sampler = HardNegativeSampler(train_ds, batch_size=16, num_pairs_per_batch=3)

print(f"\nHard Negative Sampler:")
print(f"Valid pairs: {len(sampler.valid_pairs)}")
print(f"First 5 valid pairs: {sampler.valid_pairs[:5]}")

# Check elk specifically
if 'elk' in sampler.class_indices:
    elk_count = len(sampler.class_indices['elk'])
    print(f"\nElk samples in sampler: {elk_count}")
else:
    print("\nERROR: Elk not found in sampler class_indices!")
    print(f"Available classes in sampler: {list(sampler.class_indices.keys())[:10]}")

# Check how many elk/moose pairs we'd sample
elk_moose_pair = ('elk', 'moose')
if elk_moose_pair in sampler.valid_pairs:
    print(f"Elk-Moose pair is valid")
else:
    print(f"ERROR: Elk-Moose pair not in valid pairs")
    
# Sample one batch and check
batch_iter = iter(sampler)
first_batch = next(batch_iter)
print(f"\nFirst batch indices: {first_batch[:5]}")

# Check what classes are in the batch
batch_classes = []
for idx in first_batch[:10]:
    _, _, _, class_name = train_ds.samples[idx]
    batch_classes.append(class_name)

print(f"Classes in first 10 of batch: {batch_classes}")
