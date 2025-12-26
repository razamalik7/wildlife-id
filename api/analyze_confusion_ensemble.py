
import torch
import sys
import io

# Force UTF-8 for Windows PowerShell output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os, json
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter

# Reuse classes
from evaluate_ensemble import WildlifeLateFusion, EvaluationDataset, load_model

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_top_confusions(confusion_matrix, class_names, top_k=10):
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and confusion_matrix[i][j] > 0:
                pairs.append(((class_names[i], class_names[j]), confusion_matrix[i][j]))
    
    # Sort by count descending
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

def main():
    print(f"🦁 ANALYZING CONFUSION MATRICES")
    
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
    
    model_b3 = load_model('oogway_b3_final_finetuned.pth', 'b3', DEVICE)
    model_cx = load_model('oogway_convnext_final_finetuned.pth', 'convnext', DEVICE)
    
    # 2. Track Confusions
    conf_b3 = np.zeros((num_classes, num_classes), dtype=int)
    conf_cx = np.zeros((num_classes, num_classes), dtype=int)
    conf_ens = np.zeros((num_classes, num_classes), dtype=int)
    
    print("\n🚀 Running Inference...")
    with torch.no_grad():
        for b3_imgs, cx_imgs, meta, labels in tqdm(loader):
            b3_imgs, cx_imgs = b3_imgs.to(DEVICE), cx_imgs.to(DEVICE)
            meta, labels = meta.to(DEVICE), labels.to(DEVICE)
            
            # Predictions
            out_b3 = model_b3(b3_imgs, meta)
            probs_b3 = torch.softmax(out_b3['species'], dim=1)
            
            out_cx = model_cx(cx_imgs, meta)
            probs_cx = torch.softmax(out_cx['species'], dim=1)
            
            # --- EXPERT OVERRIDE LOGIC (Matches evaluate_ensemble.py) ---
            
            # 1. Get initial classes
            _, idx_b3 = torch.max(probs_b3, 1)
            _, idx_cx = torch.max(probs_cx, 1)
            
            # 2. Hardcoded Overrides
            EXPERT_OVERRIDES = {
                ('striped_skunk', 'wild_boar'),
                ('sea_otter', 'harbor_seal'),
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
            
            probs_ens = torch.zeros_like(probs_cx)
            for i in range(len(labels)):
                idx_b3_i = idx_b3[i].item()
                idx_cx_i = idx_cx[i].item()
                
                pred_b3_name = dataset.classes[idx_b3_i]
                pred_cx_name = dataset.classes[idx_cx_i]
                
                if (pred_cx_name, pred_b3_name) in EXPERT_OVERRIDES:
                    probs_ens[i] = (0.7 * probs_b3[i]) + (0.3 * probs_cx[i])
                elif pred_b3_name in B3_SUPERIOR_CLASSES:
                    probs_ens[i] = (0.6 * probs_b3[i]) + (0.4 * probs_cx[i])
                else:
                    probs_ens[i] = (0.4 * probs_b3[i]) + (0.6 * probs_cx[i])
            
            # --- END OF LOGIC ---
            
            _, pred_b3 = torch.max(probs_b3, 1)
            _, pred_cx = torch.max(probs_cx, 1)
            _, pred_ens = torch.max(probs_ens, 1)
            
            # Update Matrices
            labels_np = labels.cpu().numpy()
            pred_b3_np = pred_b3.cpu().numpy()
            pred_cx_np = pred_cx.cpu().numpy()
            pred_ens_np = pred_ens.cpu().numpy()
            
            for i in range(len(labels)):
                true = labels_np[i]
                conf_b3[true][pred_b3_np[i]] += 1
                conf_cx[true][pred_cx_np[i]] += 1
                conf_ens[true][pred_ens_np[i]] += 1

    # 3. Print Reports
    models_data = [
        ("EfficientNet-B3", conf_b3),
        ("ConvNeXt-Tiny", conf_cx),
        ("Smart Ensemble", conf_ens)
    ]
    
    for name, matrix in models_data:
        print(f"\n{'='*60}")
        print(f"📉 TOP 10 CONFUSIONS: {name}")
        print(f"{'='*60}")
        top_errors = get_top_confusions(matrix, class_names)
        
        print(f"{'TRUE SPECIES':<30} -> {'PREDICTED':<30} | COUNT")
        print("-" * 70)
        for (true, pred), count in top_errors:
            print(f"{true:<30} -> {pred:<30} | {count}")
            
    # 4. Compare improvement on top error
    print(f"\n{'='*60}")
    print("🏆 IMPACT ANALYSIS")
    top_bad_pair = get_top_confusions(conf_b3, class_names, 1)[0]
    pair = top_bad_pair[0]
    
    b3_err = conf_b3[class_names.index(pair[0])][class_names.index(pair[1])]
    cx_err = conf_cx[class_names.index(pair[0])][class_names.index(pair[1])]
    ens_err = conf_ens[class_names.index(pair[0])][class_names.index(pair[1])]
    
    print(f"Top B3 Error: {pair[0]} -> {pair[1]}")
    print(f"B3 Count:  {b3_err}")
    print(f"CX Count:  {cx_err}")
    print(f"Ens Count: {ens_err}")

    # Write to file directly to avoid console encoding hell
    with open('final_report.txt', 'w', encoding='utf-8') as f:
        # 3. BIDIRECTIONAL PAIR ANALYSIS
        f.write(f"\n{'='*80}\n")
        f.write(f"🔄 TOP 50 BIDIRECTIONAL CONFUSION PAIRS (Best Ensemble)\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'PAIR (A <-> B)':<40} | {'TOTAL':<5} | {'LEAN':<30}\n")
        f.write("-" * 80 + "\n")
        
        # Store pairs as tuple(sorted_names) to aggregate A->B and B->A
        pair_stats = defaultdict(lambda: {'A_to_B': 0, 'B_to_A': 0, 'total': 0})
        
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j: continue
                
                count = conf_ens[i][j]
                if count > 0:
                    name_i = class_names[i]
                    name_j = class_names[j]
                    
                    # Sort to ensure (A, B) is same as (B, A)
                    pair_key = tuple(sorted([name_i, name_j]))
                    
                    pair_stats[pair_key]['total'] += count
                    
                    # Determine direction
                    if name_i == pair_key[0]: 
                        pair_stats[pair_key]['A_to_B'] += count # A is Truth, predicted B
                    else:
                        pair_stats[pair_key]['B_to_A'] += count # B is Truth, predicted A
                        
        # Sort by total confusion
        sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for i, (pair, stats) in enumerate(sorted_pairs[:50]):
            A, B = pair
            total = stats['total']
            a2b = stats['A_to_B'] # True A -> Pred B
            b2a = stats['B_to_A'] # True B -> Pred A
            
            # Calculate lean
            if a2b > b2a:
                lean_pct = (a2b / total) * 100
                lean_str = f"-> {B} ({lean_pct:.0f}%)"
            elif b2a > a2b:
                lean_pct = (b2a / total) * 100
                lean_str = f"<- {A} ({lean_pct:.0f}%)" 
            else:
                lean_str = "Balanced (50/50)"
                
            f.write(f"{A:<19} <-> {B:<18} | {total:<5} | {lean_str:<30} (A->B: {a2b}, B->A: {b2a})\n")
            
    print("Report written to final_report.txt")


if __name__ == "__main__":
    main()

