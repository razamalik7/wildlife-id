
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# Import existing classes
from evaluate_ensemble import WildlifeLateFusion, EvaluationDataset, load_model

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🦁 FINDING EXPERT OVERRIDES")
    print(f"Searching for cases where B3 identifies a species that ConvNeXt misses...")
    
    # 1. Dataset & Models
    tf_b3 = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tf_cx = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dir = os.path.join(DATA_DIR, 'val')
    dataset = EvaluationDataset(val_dir, tf_b3, tf_cx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = dataset.classes
    
    model_b3 = load_model('oogway_b3_best.pth', 'b3', DEVICE)
    model_cx = load_model('oogway_convnext_final.pth', 'convnext', DEVICE)
    
    # 2. Track Differences
    # We want to find: Which True Classes does B3 get right, but ConvNeXt gets wrong?
    # Store: {class_name: {'b3_correct': 0, 'cx_correct': 0, 'total': 0}}
    class_stats = defaultdict(lambda: {'b3_correct': 0, 'cx_correct': 0, 'total': 0})
    
    # Specific Confusion Analysis
    # Store: {(True, Pred_CX): count} where B3 was Correct
    cx_errors_b3_fixed = defaultdict(int)
    
    print("\n🚀 Running Inference...")
    with torch.no_grad():
        for b3_imgs, cx_imgs, meta, labels in tqdm(loader):
            b3_imgs, cx_imgs = b3_imgs.to(DEVICE), cx_imgs.to(DEVICE)
            meta, labels = meta.to(DEVICE), labels.to(DEVICE)
            
            # B3
            out_b3 = model_b3(b3_imgs, meta)
            _, pred_b3 = torch.max(out_b3['species'], 1)
            
            # CX
            out_cx = model_cx(cx_imgs, meta)
            _, pred_cx = torch.max(out_cx['species'], 1)
            
            # Analysis
            for i in range(len(labels)):
                label_idx = labels[i].item()
                true_name = class_names[label_idx]
                
                b3_is_correct = (pred_b3[i] == labels[i])
                cx_is_correct = (pred_cx[i] == labels[i])
                
                class_stats[true_name]['total'] += 1
                if b3_is_correct: class_stats[true_name]['b3_correct'] += 1
                if cx_is_correct: class_stats[true_name]['cx_correct'] += 1
                
                # Case: CX failed, but B3 succeeded
                if b3_is_correct and not cx_is_correct:
                    cx_pred_name = class_names[pred_cx[i].item()]
                    # Record the specific confusion ConvNeXt made
                    cx_errors_b3_fixed[(true_name, cx_pred_name)] += 1

    # 3. Report Results
    print(f"\n{'='*80}")
    print(f"🕵️  EXPERT OVERRIDES DETECTED")
    print(f"When ConvNeXt makes mistake X -> Y, but B3 gets it right:")
    print(f"{'='*80}")
    print(f"{'TRUE SPECIES':<25} | {'CONVNEXT WRONG PRED':<25} | {'COUNT (B3 SAVED)':<10}")
    print("-" * 80)
    
    # Sort by count
    sorted_fixes = sorted(cx_errors_b3_fixed.items(), key=lambda x: x[1], reverse=True)
    
    relevant_fixes = []
    for (true, bad_pred), count in sorted_fixes:
        if count >= 2: # Filter out one-offs
            print(f"{true:<25} | {bad_pred:<25} | {count:<10}")
            relevant_fixes.append(((true, bad_pred), count))
            
    print(f"\n{'='*80}")
    print("📈 CLASSES WHERE B3 IS BETTER")
    print(f"{'CLASS':<25} | {'B3 ACC':<8} | {'CX ACC':<8} | {'DIFF':<8}")
    print("-" * 80)
    
    b3_better_counts = 0
    for cls in sorted(class_names):
        stats = class_stats[cls]
        if stats['total'] == 0: continue
        
        b3_acc = 100 * stats['b3_correct'] / stats['total']
        cx_acc = 100 * stats['cx_correct'] / stats['total']
        
        if b3_acc > cx_acc:
            diff = b3_acc - cx_acc
            print(f"{cls:<25} | {b3_acc:<6.1f}% | {cx_acc:<6.1f}% | +{diff:<6.1f}%")
            b3_better_counts += 1
            
    print(f"Total classes where B3 is superior: {b3_better_counts}")

if __name__ == "__main__":
    main()
