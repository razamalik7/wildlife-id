
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
from evaluate_ensemble import WildlifeLateFusion, EvaluationDataset, load_model

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🦁 ANALYZING PER-CLASS PERFORMANCE")
    
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
    num_classes = len(class_names)
    
    model_b3 = load_model('oogway_b3_best.pth', 'b3', DEVICE)
    model_cx = load_model('oogway_convnext_final.pth', 'convnext', DEVICE)
    
    # Track stats
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Hardcoded overrides (matches final engine)
    EXPERT_OVERRIDES = {
        ('striped_skunk', 'wild_boar'),
        ('sea_otter', 'harbor_seal'),
        ('coyote', 'gray_wolf'),
        ('american_alligator', 'american_crocodile'),
        ('nile_monitor', 'argentine_black_and_white_tegu'),
        ('white-tailed_deer', 'moose'),
        ('california_sea_lion', 'northern_elephant_seal'),
        ('burmese_python', 'western_diamondback_rattlesnake')
    }
    
    B3_SUPERIOR_CLASSES = {
        'nile_monitor', 'yellow-bellied_marmot', 'harbor_seal', 
        'virginia_opossum', 'thinhorn_sheep', 'spotted_salamander',
        'western_diamondback_rattlesnake', 'mountain_lion', 
        'american_marten', 'common_box_turtle', 'great_horned_owl'
    }

    print("\n🚀 Running Evaluation...")
    with torch.no_grad():
        for b3_imgs, cx_imgs, meta, labels in tqdm(loader):
            b3_imgs, cx_imgs = b3_imgs.to(DEVICE), cx_imgs.to(DEVICE)
            meta, labels = meta.to(DEVICE), labels.to(DEVICE)
            
            out_b3 = model_b3(b3_imgs, meta)
            probs_b3 = torch.softmax(out_b3['species'], dim=1)
            
            out_cx = model_cx(cx_imgs, meta)
            probs_cx = torch.softmax(out_cx['species'], dim=1)
            
            # --- Ensembling Logic ---
            probs_ens = torch.zeros_like(probs_cx)
            
            _, idx_b3 = torch.max(probs_b3, 1)
            _, idx_cx = torch.max(probs_cx, 1)
            
            for i in range(len(labels)):
                idx_b3_i = idx_b3[i].item()
                idx_cx_i = idx_cx[i].item()
                
                pred_b3_name = class_names[idx_b3_i]
                pred_cx_name = class_names[idx_cx_i]
                
                if (pred_cx_name, pred_b3_name) in EXPERT_OVERRIDES:
                    probs_ens[i] = (0.7 * probs_b3[i]) + (0.3 * probs_cx[i])
                elif pred_b3_name in B3_SUPERIOR_CLASSES:
                    probs_ens[i] = (0.6 * probs_b3[i]) + (0.4 * probs_cx[i])
                else:
                    probs_ens[i] = (0.4 * probs_b3[i]) + (0.6 * probs_cx[i])
            
            _, pred_ens = torch.max(probs_ens, 1)
            
            # Update stats
            labels_np = labels.cpu().numpy()
            pred_np = pred_ens.cpu().numpy()
            
            for i in range(len(labels)):
                lbl = labels_np[i]
                pred = pred_np[i]
                class_total[lbl] += 1
                confusion_matrix[lbl][pred] += 1
                if pred == lbl:
                    class_correct[lbl] += 1
                    
    # Report
    print(f"\n{'='*60}")
    print(f"📊 LOWEST ACCURACY CLASSES (Ideally > 85%)")
    print(f"{'='*60}")
    print(f"{'CLASS':<30} | {'ACCURACY':<10} | {'SAMPLES':<8}")
    print("-" * 60)
    
    class_accs = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = (class_correct[i] / class_total[i]) * 100
            class_accs.append((class_names[i], acc, int(class_total[i])))
            
    # Sort by Accuracy Ascending (Worst first)
    sorted_accs = sorted(class_accs, key=lambda x: x[1])
    
    for name, acc, count in sorted_accs[:30]: # Bottom 30
        print(f"{name:<30} | {acc:<6.1f}%    | {count:<8}")
        
    print(f"\n{'='*60}")
    
    # Save recommended targets based on ACCURACY, not just error count
    targets = {}
    
    # 1. Primary Targets (Weak Accuracy)
    print("🎯 Calculated Targets:")
    for i, (name, acc, count) in enumerate(sorted_accs):
        if acc < 95.0: # Threshold for "Needs Improvement"
            deficit = 95.0 - acc
            target = int(deficit * 30) # 30 images per missing % point
            target = min(target, 600) # Cap at 600
            target = max(target, 100) # Min 100
            targets[name] = target
            
    # 2. Safety Partners (Confusion Matrix)
    SAFETY_AMOUNT = 150
    print("\n🛡️ Safety Partners Added:")
    
    for name in list(targets.keys()):
        idx = class_names.index(name)
        
        # Find who this class is confused with Most
        # Row 'idx' in confusion matrix = Truth is 'name', Predicted is 'col'
        # sort row by count descending
        row = confusion_matrix[idx]
        # set diagonal to -1 to ignore
        row[idx] = -1 
        
        confused_idx = np.argmax(row)
        confused_count = row[confused_idx]
        
        if confused_count > 0:
            partner_name = class_names[confused_idx]
            
            # If partner not in targets, add it
            if partner_name not in targets:
                targets[partner_name] = SAFETY_AMOUNT
                print(f"   + {partner_name:<25} (Safety for {name}, {confused_count} errs)")
            
            # If partner is in targets but very low (e.g. < 150), boost it
            elif targets[partner_name] < SAFETY_AMOUNT:
                 targets[partner_name] = SAFETY_AMOUNT
                 print(f"   ^ {partner_name:<25} (Boosted for {name})")

    print(f"\nTotal Targets: {len(targets)} classes")
    
    import json
    with open('scrape_targets_refined.json', 'w') as f:
        json.dump(targets, f, indent=2)

if __name__ == "__main__":
    main()
