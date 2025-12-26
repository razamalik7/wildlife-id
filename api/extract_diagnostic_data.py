"""
Extract diagnostic data and save to JSON for analysis
"""
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from train_oogway import WildlifeGeoDataset, WildlifeLateFusion
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load validation dataset
    val_transform = transforms.Compose([
        transforms.Resize(330),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_ds = WildlifeGeoDataset('./training_data_cropped/val', val_transform, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Load Oogway model
    ckpt = torch.load('oogway_b3_best.pth', map_location='cpu')
    model = WildlifeLateFusion(
        len(val_ds.classes), 
        len(val_ds.families), 
        len(val_ds.tax_classes), 
        model_type='b3'
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Evaluate
    species_correct = 0
    family_correct = 0
    class_correct = 0
    total = 0

    num_species = len(val_ds.classes)
    confusion_matrix = np.zeros((num_species, num_species), dtype=int)

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
            imgs, meta, species_labels, family_labels, class_labels = batch_data
            imgs = imgs.to(DEVICE)
            meta = meta.to(DEVICE)
            species_labels = species_labels.to(DEVICE)
            family_labels = family_labels.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            
            outputs = model(imgs, meta)
            
            _, species_preds = torch.max(outputs['species'], 1)
            _, family_preds = torch.max(outputs['family'], 1)
            _, class_preds = torch.max(outputs['class'], 1)
            
            species_correct += (species_preds == species_labels).sum().item()
            family_correct += (family_preds == family_labels).sum().item()
            class_correct += (class_preds == class_labels).sum().item()
            total += species_labels.size(0)
            
            for true, pred in zip(species_labels.cpu().numpy(), species_preds.cpu().numpy()):
                confusion_matrix[true, pred] += 1

    # Compute all metrics
    results = {
        'accuracies': {
            'species': species_correct / total,
            'family': family_correct / total,
            'class': class_correct / total
        },
        'per_class_accuracy': [],
        'bidirectional_confusion_pairs': [],
        'confusion_matrix': confusion_matrix.tolist()
    }

    # Per-class accuracy
    for i, class_name in enumerate(val_ds.classes):
        total_samples = confusion_matrix[i].sum()
        if total_samples > 0:
            correct_samples = confusion_matrix[i, i]
            acc = correct_samples / total_samples
            results['per_class_accuracy'].append({
                'species': class_name,
                'accuracy': acc,
                'correct': int(correct_samples),
                'total': int(total_samples)
            })

    results['per_class_accuracy'].sort(key=lambda x: x['accuracy'])

    # Bidirectional pairs
    for i in range(num_species):
        for j in range(i+1, num_species):
            errors_ij = confusion_matrix[i, j]
            errors_ji = confusion_matrix[j, i]
            total_errors = errors_ij + errors_ji
            
            if total_errors > 0:
                results['bidirectional_confusion_pairs'].append({
                    'pair': sorted([val_ds.classes[i], val_ds.classes[j]]),
                    'total_errors': int(total_errors),
                    'directional': {
                        f'{val_ds.classes[i]}_to_{val_ds.classes[j]}': int(errors_ij),
                        f'{val_ds.classes[j]}_to_{val_ds.classes[i]}': int(errors_ji)
                    }
                })

    results['bidirectional_confusion_pairs'].sort(key=lambda x: x['total_errors'], reverse=True)

    # Save to JSON
    with open('oogway_diagnostic_data.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Saved diagnostic data to oogway_diagnostic_data.json")
    print(f"Species Accuracy: {results['accuracies']['species']:.4f}")
    print(f"Family Accuracy: {results['accuracies']['family']:.4f}")
    print(f"Class Accuracy: {results['accuracies']['class']:.4f}")
