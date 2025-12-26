"""
Diagnose Oogway performance issue
Compare with Grandmaster to understand the accuracy gap
"""
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

    print("Loading validation dataset...")
    val_ds = WildlifeGeoDataset('./training_data_cropped/val', val_transform, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)  # num_workers=0 to avoid multiprocessing

    print(f"Validation samples: {len(val_ds)}")
    print(f"Num species: {len(val_ds.classes)}")
    print(f"Num families: {len(val_ds.families)}")
    print(f"Num tax classes: {len(val_ds.tax_classes)}")

    # Load Oogway model
    print("\nLoading Oogway B3 model...")
    ckpt = torch.load('oogway_b3_best.pth', map_location='cpu')
    model = WildlifeLateFusion(
        len(val_ds.classes), 
        len(val_ds.families), 
        len(val_ds.tax_classes), 
        model_type='b3'
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Model from epoch: {ckpt['epoch']}")
    print(f"Saved accuracy: {ckpt['accuracy']:.4f} ({ckpt['accuracy']*100:.2f}%)")

    # Evaluate with confusion matrix
    print("\nEvaluating...")
    species_correct = 0
    family_correct = 0
    class_correct = 0
    total = 0

    # Confusion matrix
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
            
            # Update confusion matrix
            for true, pred in zip(species_labels.cpu().numpy(), species_preds.cpu().numpy()):
                confusion_matrix[true, pred] += 1

    species_acc = species_correct/total
    family_acc = family_correct/total
    class_acc = class_correct/total

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Species Accuracy: {species_acc:.4f} ({species_acc*100:.2f}%)")
    print(f"Family Accuracy: {family_acc:.4f} ({family_acc*100:.2f}%)")
    print(f"Class Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%)")

    # Per-class accuracy
    print("\nPer-Class Analysis:")
    class_accuracies = []
    for i, class_name in enumerate(val_ds.classes):
        total_samples = confusion_matrix[i].sum()
        if total_samples > 0:
            correct_samples = confusion_matrix[i, i]
            acc = correct_samples / total_samples
            class_accuracies.append((class_name, acc, total_samples, correct_samples))

    class_accuracies.sort(key=lambda x: x[1])

    print("\nWorst 15 Species:")
    for name, acc, total_s, correct_s in class_accuracies[:15]:
        print(f"  {name:30s}: {acc:6.2%} ({correct_s:3d}/{total_s:3d})")

    print("\nBest 10 Species:")
    for name, acc, total_s, correct_s in class_accuracies[-10:]:
        print(f"  {name:30s}: {acc:6.2%} ({correct_s:3d}/{total_s:3d})")

    # Top confused pairs - BIDIRECTIONAL analysis
    print("\n" + "="*70)
    print("TOP 25 CONFUSED PAIRS (Bidirectional Analysis)")
    print("="*70)
    
    # Aggregate bidirectional errors
    bidirectional_pairs = {}
    for i in range(num_species):
        for j in range(i+1, num_species):  # Only look at upper triangle
            errors_ij = confusion_matrix[i, j]
            errors_ji = confusion_matrix[j, i]
            total_errors = errors_ij + errors_ji
            
            if total_errors > 0:
                pair_key = tuple(sorted([val_ds.classes[i], val_ds.classes[j]]))
                bidirectional_pairs[pair_key] = {
                    'total': total_errors,
                    'forward': (val_ds.classes[i], val_ds.classes[j], errors_ij),
                    'backward': (val_ds.classes[j], val_ds.classes[i], errors_ji)
                }
    
    # Sort by total errors
    sorted_pairs = sorted(bidirectional_pairs.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for rank, (pair, data) in enumerate(sorted_pairs[:25], 1):
        a, b = pair
        fwd_a, fwd_b, fwd_err = data['forward']
        bck_a, bck_b, bck_err = data['backward']
        total = data['total']
        
        # Determine directionality
        if fwd_err > bck_err * 1.5:
            bias = f"→ (biased {fwd_a}→{fwd_b})"
        elif bck_err > fwd_err * 1.5:
            bias = f"← (biased {bck_a}→{bck_b})"
        else:
            bias = "↔ (symmetric)"
        
        print(f"  {rank:2d}. {a:25s} ↔ {b:25s}: {total:3d} errors {bias}")
        print(f"      {fwd_a:25s} → {fwd_b:25s}: {fwd_err:3d}")
        print(f"      {bck_a:25s} → {bck_b:25s}: {bck_err:3d}")

    # Compare with Grandmaster
    print("\n" + "="*70)
    print("COMPARISON WITH GRANDMASTER")
    print("="*70)
    print(f"Grandmaster B3: 76.50%")
    print(f"Oogway B3:      {species_acc*100:.2f}%")
    print(f"Difference:     {(species_acc - 0.765)*100:+.2f}%")
    print(f"\nFamily Accuracy: {family_acc*100:.2f}%")
    print(f"Class Accuracy:  {class_acc*100:.2f}%")

    if species_acc < 0.765:
        print("\n⚠️  OOGWAY IS UNDERPERFORMING GRANDMASTER!")
        print("\nPossible causes:")
        print("  1. Multi-task learning causing gradient conflicts")
        print("  2. Loss weights too aggressive (alpha_family=0.3, alpha_class=0.1)")
        print("  3. Over-regularization from 8+ techniques")
        print("  4. Hierarchical focal loss interfering with species learning")
        print("  5. Hard negative mining biasing towards confused pairs")
        print("\nDEEPER ISSUES TO INVESTIGATE:")
        print("  - Compare confusion pairs with Grandmaster's top 25")
        print("  - Check if multi-task improved family/class predictions at species cost")
        print("  - Analyze which regularization techniques help vs hurt")
    else:
        print("\n✅ OOGWAY MATCHES OR EXCEEDS GRANDMASTER!")
